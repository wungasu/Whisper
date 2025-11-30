import os
import argparse
import sys
import asyncio
import sqlite3
import hashlib
import secrets
import time
from typing import List, Optional, Tuple
from uuid import uuid4
import gradio as gr
from gradio_i18n import Translate, gettext as _
import yaml

from modules.utils.paths import (FASTER_WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, OUTPUT_DIR, WHISPER_MODELS_DIR,
                                 INSANELY_FAST_WHISPER_MODELS_DIR, NLLB_MODELS_DIR, DEFAULT_PARAMETERS_CONFIG_PATH,
                                 UVR_MODELS_DIR, I18N_YAML_PATH, KNOWLEDGE_BASE_DIR, RAG_STORE_DIR)
from modules.utils.files_manager import load_yaml, MEDIA_EXTENSION
from modules.whisper.whisper_factory import WhisperFactory
from modules.translation.nllb_inference import NLLBInference
from modules.ui.htmls import *
from modules.utils.cli_manager import str2bool
from modules.utils.youtube_manager import get_ytmetas
from modules.translation.deepl_api import DeepLAPI
from modules.whisper.data_classes import *
from modules.utils.logger import get_logger
from modules.face_search.service import FaceSearchService
from modules.rag.text_corrector import TextCorrectionRAG
from modules.rag.temp_chat import TemporaryRAGChatService


logger = get_logger()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AUTH_DB_PATH = os.path.join(BASE_DIR, "auth.db")
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "88888888"

# 修复 Windows 上 asyncio 的连接重置警告
def handle_exception(loop, context):
    """处理 asyncio 异常，避免 Windows 连接重置错误显示"""
    exception = context.get('exception')
    if isinstance(exception, ConnectionResetError):
        # Windows 上的连接重置错误通常是正常的，可以忽略
        if exception.errno == 10054:  # WinError 10054
            # 这是一个常见的 Windows 网络问题，不影响应用运行
            pass
        else:
            # 其他连接重置错误仍然记录
            logger.debug(f"Asyncio connection error: {exception}")
    elif isinstance(exception, OSError):
        # 其他 OSError 也可能正常，不记录
        if hasattr(exception, 'winerror') and exception.winerror == 10054:
            pass
        else:
            logger.debug(f"Asyncio OSError: {exception}")
    else:
        # 其他异常正常处理
        logger.warning(f"Asyncio exception: {context}")

# 在 Windows 上设置 asyncio 异常处理器
if sys.platform == 'win32':
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果循环已在运行，设置异常处理器
            loop.set_exception_handler(handle_exception)
        else:
            # 如果循环未运行，在下次创建时设置
            def create_loop():
                loop = asyncio.new_event_loop()
                loop.set_exception_handler(handle_exception)
                return loop
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        # 如果设置失败，不影响应用运行
        pass

class App:
    def __init__(self, args):
        self.args = args
        # Check every 1 hour (3600) for cached files and delete them if older than 1 day (86400)
        self.app = gr.Blocks(css=CSS, theme=self.args.theme, delete_cache=(3600, 86400))
        self.whisper_inf = WhisperFactory.create_whisper_inference(
            whisper_type=self.args.whisper_type,
            whisper_model_dir=self.args.whisper_model_dir,
            faster_whisper_model_dir=self.args.faster_whisper_model_dir,
            insanely_fast_whisper_model_dir=self.args.insanely_fast_whisper_model_dir,
            uvr_model_dir=self.args.uvr_model_dir,
            output_dir=self.args.output_dir,
        )
        self.nllb_inf = NLLBInference(
            model_dir=self.args.nllb_model_dir,
            output_dir=os.path.join(self.args.output_dir, "translations")
        )
        self.deepl_api = DeepLAPI(
            output_dir=os.path.join(self.args.output_dir, "translations")
        )
        try:
            self.text_corrector = TextCorrectionRAG(
                knowledge_dir=self.args.rag_kb_dir,
                persist_dir=self.args.rag_store_dir,
                embedding_model=self.args.rag_embedding_model,
            )
            self.whisper_inf.register_text_corrector(self.text_corrector)
        except Exception as exc:
            logger.warning(f"初始化 RAG 文本纠错失败，将禁用该功能：{exc}")
            self.text_corrector = None
        self.temp_chat_service = TemporaryRAGChatService(
            embedding_model=self.args.rag_embedding_model
        )
        self.auth_db_path = self.args.auth_db_path or DEFAULT_AUTH_DB_PATH
        self.default_admin_username = self.args.default_admin_username or DEFAULT_ADMIN_USERNAME
        self.default_admin_password = self.args.default_admin_password or DEFAULT_ADMIN_PASSWORD
        self.i18n = load_yaml(I18N_YAML_PATH)
        self.default_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
        try:
            if isinstance(self.default_params, dict):
                whisper_defaults = self.default_params.setdefault("whisper", {})
                whisper_defaults.pop("initial_prompt", None)
        except Exception:
            pass

        self.init_db()
        logger.info(f"Use \"{self.args.whisper_type}\" implementation\n"
                    f"Device \"{self.whisper_inf.device}\" is detected")

    def _ensure_auth_storage(self):
        db_dir = os.path.dirname(self.auth_db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    def _connect_auth_db(self):
        conn = sqlite3.connect(self.auth_db_path, timeout=5, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _hash_password(password: str, salt_hex: Optional[str] = None) -> str:
        if not password:
            raise ValueError("Password is required.")
        salt_bytes = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
        hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 120000)
        return f"{salt_bytes.hex()}${hashed.hex()}"

    @staticmethod
    def _verify_password(password: str, stored_value: str) -> bool:
        if not password or not stored_value:
            return False
        try:
            salt_hex, hashed_hex = stored_value.split("$", 1)
        except ValueError:
            return False
        computed_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            120000
        ).hex()
        return secrets.compare_digest(hashed_hex, computed_hash)

    def init_db(self):
        self._ensure_auth_storage()
        with self._connect_auth_db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    status TEXT NOT NULL DEFAULT 'pending'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)
            """)
            cur = conn.execute("SELECT id FROM users WHERE username=?", (self.default_admin_username,))
            if not cur.fetchone():
                hashed_pwd = self._hash_password(self.default_admin_password)
                conn.execute(
                    "INSERT INTO users (username, password, role, status) VALUES (?, ?, 'admin', 'active')",
                    (self.default_admin_username, hashed_pwd)
                )
                logger.info(f"已创建默认管理员账号：{self.default_admin_username}")

    def register_user(self, username: Optional[str], password: Optional[str]):
        username = (username or "").strip()
        password = password or ""
        if not username or not password:
            return False, "用户名和密码均不能为空。"
        if len(password) < 6:
            return False, "密码长度至少为 6 位。"
        try:
            hashed_pwd = self._hash_password(password)
            with self._connect_auth_db() as conn:
                conn.execute(
                    "INSERT INTO users (username, password, role, status) VALUES (?, ?, 'user', 'pending')",
                    (username, hashed_pwd)
                )
            return True, f"注册成功！{username} 已提交审核，请等待管理员批准。"
        except sqlite3.IntegrityError:
            return False, "该用户名已存在，请更换后再试。"
        except Exception as exc:
            logger.error(f"注册用户失败：{exc}", exc_info=True)
            return False, f"注册失败：{exc}"

    def login_user(self, username: Optional[str], password: Optional[str]):
        username = (username or "").strip()
        password = password or ""
        if not username or not password:
            return False, None, "请输入用户名和密码。"
        try:
            with self._connect_auth_db() as conn:
                row = conn.execute(
                    "SELECT password, role, status FROM users WHERE username=?",
                    (username,)
                ).fetchone()
        except Exception as exc:
            logger.error(f"登录查询失败：{exc}", exc_info=True)
            return False, None, "登录失败，请稍后再试。"

        if not row or not self._verify_password(password, row["password"]):
            return False, None, "用户名或密码不正确。"
        if row["status"] != "active":
            return False, None, "该账号尚未通过管理员审核。"
        return True, row["role"], f"欢迎，{username}！"

    def get_pending_users(self):
        try:
            with self._connect_auth_db() as conn:
                rows = conn.execute(
                    "SELECT username FROM users WHERE status='pending' ORDER BY username ASC"
                ).fetchall()
            return [row["username"] for row in rows]
        except Exception as exc:
            logger.error(f"获取待审核用户失败：{exc}", exc_info=True)
            return []

    def approve_user(self, username: Optional[str]):
        username = (username or "").strip()
        if not username:
            return False, "请选择需要批准的用户名。"
        try:
            with self._connect_auth_db() as conn:
                cur = conn.execute(
                    "UPDATE users SET status='active' WHERE username=? AND status='pending'",
                    (username,)
                )
                if cur.rowcount == 0:
                    return False, "未找到该用户或该用户已被审核。"
            return True, f"用户 {username} 已成功激活。"
        except Exception as exc:
            logger.error(f"批准用户失败：{exc}", exc_info=True)
            return False, f"审批失败：{exc}"

    def get_all_users(self):
        """获取所有用户列表（不包括当前主账号admin）"""
        try:
            with self._connect_auth_db() as conn:
                rows = conn.execute(
                    "SELECT username, role, status FROM users WHERE username != ? ORDER BY username ASC",
                    (self.default_admin_username,)
                ).fetchall()
            return [
                {"username": row["username"], "role": row["role"], "status": row["status"]}
                for row in rows
            ]
        except Exception as exc:
            logger.error(f"获取用户列表失败：{exc}", exc_info=True)
            return []

    def grant_admin_role(self, target_username: Optional[str], current_username: Optional[str]):
        """赋予用户管理员权限（仅主账号admin可操作）"""
        target_username = (target_username or "").strip()
        current_username = (current_username or "").strip()
        
        if not target_username:
            return False, "请选择要赋予管理员权限的用户。"
        
        # 检查当前用户是否是主账号admin
        if current_username != self.default_admin_username:
            return False, "只有主账号可以赋予管理员权限。"
        
        if target_username == self.default_admin_username:
            return False, "不能修改主账号的权限。"
        
        try:
            with self._connect_auth_db() as conn:
                # 检查目标用户是否存在
                cur = conn.execute(
                    "SELECT id FROM users WHERE username=?",
                    (target_username,)
                )
                if not cur.fetchone():
                    return False, f"用户 {target_username} 不存在。"
                
                # 更新用户角色为admin
                cur = conn.execute(
                    "UPDATE users SET role='admin' WHERE username=?",
                    (target_username,)
                )
                if cur.rowcount == 0:
                    return False, f"更新用户 {target_username} 的权限失败。"
            return True, f"已成功将 {target_username} 设置为管理员。"
        except Exception as exc:
            logger.error(f"赋予管理员权限失败：{exc}", exc_info=True)
            return False, f"操作失败：{exc}"

    def revoke_admin_role(self, target_username: Optional[str], current_username: Optional[str]):
        """撤销用户管理员权限（仅主账号admin可操作）"""
        target_username = (target_username or "").strip()
        current_username = (current_username or "").strip()
        
        if not target_username:
            return False, "请选择要撤销管理员权限的用户。"
        
        # 检查当前用户是否是主账号admin
        if current_username != self.default_admin_username:
            return False, "只有主账号可以撤销管理员权限。"
        
        if target_username == self.default_admin_username:
            return False, "不能修改主账号的权限。"
        
        try:
            with self._connect_auth_db() as conn:
                # 检查目标用户是否存在
                cur = conn.execute(
                    "SELECT id FROM users WHERE username=?",
                    (target_username,)
                )
                if not cur.fetchone():
                    return False, f"用户 {target_username} 不存在。"
                
                # 更新用户角色为user
                cur = conn.execute(
                    "UPDATE users SET role='user' WHERE username=?",
                    (target_username,)
                )
                if cur.rowcount == 0:
                    return False, f"更新用户 {target_username} 的权限失败。"
            return True, f"已成功将 {target_username} 的管理员权限撤销。"
        except Exception as exc:
            logger.error(f"撤销管理员权限失败：{exc}", exc_info=True)
            return False, f"操作失败：{exc}"

    def create_pipeline_inputs(self):
        whisper_params = self.default_params["whisper"]
        vad_params = self.default_params["vad"]
        diarization_params = self.default_params["diarization"]
        uvr_params = self.default_params["bgm_separation"]

        with gr.Row():
            dd_model = gr.Dropdown(choices=self.whisper_inf.available_models, value=whisper_params["model_size"],
                                   label=_("Model"), allow_custom_value=True)
            dd_lang = gr.Dropdown(choices=self.whisper_inf.available_langs + [AUTOMATIC_DETECTION],
                                  value=AUTOMATIC_DETECTION if whisper_params["lang"] == AUTOMATIC_DETECTION.unwrap()
                                  else whisper_params["lang"], label=_("Language"))
            dd_file_format = gr.Dropdown(choices=["SRT", "WebVTT", "txt", "LRC"], value=whisper_params["file_format"], label=_("File Format"))
        # with gr.Row():
        #     cb_translate = gr.Checkbox(value=whisper_params["is_translate"], label=_("Translate to English?"),
        #                                interactive=True)
        # 创建一个占位符，因为 cb_translate 已被注释掉
        cb_translate = gr.Checkbox(value=False, label=_("Translate to English?"), visible=False)
        
        with gr.Row():
            cb_timestamp = gr.Checkbox(value=whisper_params["add_timestamp"],
                                       label=_("Add a timestamp to the end of the filename"),
                                       interactive=True)

        with gr.Accordion(_("Advanced Parameters"), open=False):
            whisper_inputs = WhisperParams.to_gradio_inputs(defaults=whisper_params, only_advanced=True,
                                                            whisper_type=self.args.whisper_type,
                                                            available_compute_types=self.whisper_inf.available_compute_types,
                                                            compute_type=self.whisper_inf.current_compute_type)
            # Keep initial prompt slot in pipeline inputs to satisfy parameter mapping, but hide from UI
            tb_initial_prompt_state = gr.State(GRADIO_NONE_STR)
            whisper_inputs.insert(8, tb_initial_prompt_state)

        with gr.Accordion(_("Background Music Remover Filter"), open=False):
            uvr_inputs = BGMSeparationParams.to_gradio_input(defaults=uvr_params,
                                                             available_models=self.whisper_inf.music_separator.available_models,
                                                             available_devices=self.whisper_inf.music_separator.available_devices,
                                                             device=self.whisper_inf.music_separator.device)

        with gr.Accordion(_("Voice Detection Filter"), open=False):
            vad_inputs = VadParams.to_gradio_inputs(defaults=vad_params)

        with gr.Accordion(_("Diarization"), open=False):
            diarization_inputs = DiarizationParams.to_gradio_inputs(defaults=diarization_params,
                                                                    available_devices=self.whisper_inf.diarizer.available_device,
                                                                    device=self.whisper_inf.diarizer.device)

        pipeline_inputs = [dd_model, dd_lang, cb_translate] + whisper_inputs + vad_inputs + diarization_inputs + uvr_inputs

        return (
            pipeline_inputs,
            dd_file_format,
            cb_timestamp
        )

    def launch(self):
        with self.app:
            lang = gr.Radio(
                choices=list(self.i18n.keys()),
                label="Language",
                interactive=True,
                visible=False,
            )
            with Translate(self.i18n):
                def _empty_user_state():
                    return {"username": None, "role": None, "authenticated": False}

                auth_state = gr.State(_empty_user_state())

                with gr.Column(elem_id="login_view_container") as login_view:
                    gr.Markdown("### 登录 Whisper WebUI")
                    gr.Markdown("如无账号，可直接输入新用户名和密码注册，注册后需管理员审批。")
                    tb_login_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
                    tb_login_password = gr.Textbox(label="密码", placeholder="请输入密码", type="password")
                    with gr.Row():
                        btn_login = gr.Button("登录", variant="primary")
                        btn_register = gr.Button("注册", variant="secondary")
                    login_feedback = gr.Markdown("")

                with gr.Column(visible=False) as main_view:
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(MARKDOWN, elem_id="md_project")
                            gr.HTML(POSTMESSAGE_FIX_JS)
                    with gr.Row():
                        user_info_box = gr.Markdown("", elem_id="current_user_info")
                        btn_logout = gr.Button("退出登录", variant="secondary")
                    with gr.Accordion("管理员审核面板", open=False, visible=False) as admin_panel:
                        gr.Markdown("仅管理员可见，用于批准新注册用户。")
                        pending_users_dropdown = gr.Dropdown(
                            label="待审核用户",
                            choices=[],
                            value=None,
                            multiselect=False,
                            interactive=True
                        )
                        with gr.Row():
                            btn_refresh_pending = gr.Button("刷新待审核列表", size="sm")
                            btn_approve_pending = gr.Button("批准选中用户", variant="primary", size="sm")
                        admin_feedback = gr.Markdown("")
                        
                        # 管理员权限管理（仅主账号admin可见）
                        with gr.Accordion("管理员权限管理", open=False, visible=False) as admin_role_panel:
                            gr.Markdown("**仅主账号 admin 可见**，用于赋予或撤销其他用户的管理员权限。")
                            all_users_dropdown = gr.Dropdown(
                                label="选择用户",
                                choices=[],
                                value=None,
                                multiselect=False,
                                interactive=True
                            )
                            user_role_display = gr.Markdown("")
                            with gr.Row():
                                btn_refresh_users = gr.Button("刷新用户列表", size="sm")
                                btn_grant_admin = gr.Button("赋予管理员权限", variant="primary", size="sm")
                                btn_revoke_admin = gr.Button("撤销管理员权限", variant="stop", size="sm")
                        admin_role_feedback = gr.Markdown("")
                    with gr.Accordion("RAG 知识库管理", open=False, visible=False) as rag_panel:
                        gr.Markdown("仅管理员可见，用于重新索引本地知识库，使 RAG 纠错能够读取最新 Markdown/Text 文档。")
                        btn_update_kb = gr.Button("更新知识库索引", variant="primary")
                        rag_update_feedback = gr.Markdown("")
                    with gr.Tabs():
                        with gr.TabItem(_("File")):  # tab1
                            with gr.Column():
                                input_file = gr.Files(type="filepath", label=_("Upload File here"), file_types=MEDIA_EXTENSION)
                                tb_input_folder = gr.Textbox(label="Input Folder Path (Optional)",
                                                             info="Optional: Specify the folder path where the input files are located, if you prefer to use local files instead of uploading them."
                                                                  " Leave this field empty if you do not wish to use a local path.",
                                                             visible=self.args.colab,
                                                             value="")
                                cb_include_subdirectory = gr.Checkbox(label="Include Subdirectory Files",
                                                                      info="When using Input Folder Path above, whether to include all files in the subdirectory or not.",
                                                                      visible=self.args.colab,
                                                                      value=False)
                                cb_save_same_dir = gr.Checkbox(label="Save outputs at same directory",
                                                               info="When using Input Folder Path above, whether to save output in the same directory as inputs or not, in addition to the original"
                                                                    " output directory.",
                                                               visible=self.args.colab,
                                                               value=True)
                            with gr.Accordion(_("转录参数设置（可选）"), open=False):
                                with gr.Column():
                                    pipeline_params, dd_file_format, cb_timestamp = self.create_pipeline_inputs()

                            with gr.Row():
                                btn_run = gr.Button(_("GENERATE SUBTITLE FILE"), variant="primary")
                            with gr.Column():
                                tb_corrected_output = gr.Textbox(
                                    label="RAG 纠错文本",
                                    lines=8,
                                    interactive=False,
                                    placeholder="运行后展示 Qwen RAG 纠错后的文本"
                                )
                            with gr.Row():
                                files_subtitles = gr.Files(label=_("Downloadable output file"), scale=3, interactive=False)
                                btn_openfolder = gr.Button('📂', scale=1)

                            with gr.Accordion(_("后处理选项（可选）"), open=False):
                                cb_convert_t2s = gr.Checkbox(
                                    label="Convert Traditional to Simplified (T->S)",
                                    value=True,
                                    info="If enabled, final subtitles will be converted to Simplified Chinese."
                                )

                            # 自动化 LLM 纠错配置（隐藏）
                            state_enable_llm = gr.State(True)
                            state_whisper_hidden = gr.State("")
                            state_rag_kb_dir = gr.State(self.args.rag_kb_dir)
                            state_ollama_base_url = gr.State("http://localhost:11434")
                            state_ollama_model = gr.State("qwen2.5:3b")
                            state_rag_top_k = gr.State(4)
                            state_rag_similarity = gr.State(0.85)
                            state_chat_payload = gr.State(None)

                            params = [
                                input_file,
                                tb_input_folder,
                                cb_include_subdirectory,
                                cb_save_same_dir,
                                dd_file_format,
                                cb_timestamp,
                                cb_convert_t2s,
                                state_enable_llm,
                                state_rag_kb_dir,
                                state_ollama_base_url,
                                state_ollama_model,
                                state_rag_top_k,
                                state_rag_similarity,
                            ]
                            params = params + pipeline_params
                            btn_run.click(fn=self.whisper_inf.transcribe_file,
                                          inputs=params,
                                          outputs=[state_whisper_hidden, tb_corrected_output, files_subtitles, state_chat_payload])
                            btn_openfolder.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)
#-------------------------------------------------------------------------------------------------------------
                        with gr.TabItem(_("访谈助手")):
                            with gr.Column():
                                chat_history = gr.Chatbot(
                                    label="临时 RAG 对话",
                                    height=420,
                                    show_copy_button=True,
                                    show_label=True,
                                )
                                tb_chat_input = gr.Textbox(
                                    label="提问或指令",
                                    placeholder="例如：请总结这段访谈的核心观点",
                                    lines=2,
                                )
                            with gr.Row():
                                slider_chat_top_k = gr.Slider(
                                    minimum=1,
                                    maximum=8,
                                    step=1,
                                    value=4,
                                    label="检索片段数量 (top_k)",
                                )
                                slider_chat_similarity = gr.Slider(
                                    minimum=0.5,
                                    maximum=0.95,
                                    step=0.05,
                                    value=0.8,
                                    label="最小相似度阈值",
                                )
                            with gr.Row():
                                btn_chat_send = gr.Button("发送", variant="primary")
                                btn_chat_clear = gr.Button("清空对话", variant="secondary")
                        gr.Markdown(
                            "完成一次转写后即可使用访谈内容进行问答；若未加载文本，则自动作为普通 AI 回答。"
                        )
                        with gr.Accordion("上传补充文档（可选）", open=False):
                            gr.Markdown("上传 `.txt` / `.md` 文件作为临时上下文，帮助 AI 回答。")
                            chat_upload = gr.Files(
                                label="选择文本或 Markdown 文件",
                                file_types=["file"],
                                type="filepath"
                            )
                            btn_load_chat_docs = gr.Button("载入到访谈助手", variant="secondary")
                            chat_upload_feedback = gr.Markdown("")

                            def _chat_with_transcript(
                                message: str,
                                history: Optional[List[List[str]]],
                                chat_payload: Optional[dict],
                                top_k: int,
                                similarity: float,
                                base_url: Optional[str],
                                model_name: str,
                            ):
                                history = history or []
                                if not message or not message.strip():
                                    gr.Warning("请输入问题或指令。")
                                    return history, ""
                                try:
                                    answer, used_context = self.temp_chat_service.generate_reply(
                                        payload=chat_payload,
                                        user_message=message.strip(),
                                        history=history,
                                        base_url=base_url,
                                        model=model_name,
                                        top_k=int(top_k),
                                        similarity_threshold=float(similarity),
                                    )
                                    if not chat_payload:
                                        gr.Info("未检测到访谈文本，将作为普通 AI 回答。")
                                    elif not used_context:
                                        gr.Info("未检索到相关访谈片段，回答基于模型常识。")
                                    updated_history = history + [[message, answer]]
                                    return updated_history, ""
                                except Exception as exc:
                                    logger.error(f"临时 RAG 对话失败: {exc}", exc_info=True)
                                    gr.Warning(f"聊天失败：{exc}")
                                    return history, message

                            def _load_chat_documents(file_list, existing_payload):
                                if not file_list:
                                    return existing_payload, "请先选择要上传的文件。"
                                resolved_paths = []
                                for item in file_list:
                                    if isinstance(item, gr.utils.NamedString):
                                        resolved_paths.append(item.name)
                                    elif isinstance(item, str):
                                        resolved_paths.append(item)
                                    elif hasattr(item, "name"):
                                        resolved_paths.append(item.name)
                                if not resolved_paths:
                                    return existing_payload, "无法解析上传文件路径。"

                                supported_ext = (".txt", ".md", ".markdown")
                                loaded_entries = []
                                failures = []

                                def _read_plain_text(path: str) -> Optional[str]:
                                    encodings = ("utf-8", "utf-8-sig", "gbk")
                                    for enc in encodings:
                                        try:
                                            with open(path, "r", encoding=enc) as f:
                                                return f.read()
                                        except UnicodeDecodeError:
                                            continue
                                        except Exception as exc:
                                            logger.warning(f"读取聊天上传文件失败：{path}, {exc}")
                                            return None
                                    return None

                                for path in resolved_paths:
                                    if not path or not os.path.exists(path):
                                        failures.append(f"{os.path.basename(path) if path else '未知文件'}：路径不存在")
                                        continue
                                    ext = os.path.splitext(path)[1].lower()
                                    if ext not in supported_ext:
                                        failures.append(f"{os.path.basename(path)}：不支持的类型（仅限 txt/md）")
                                        continue
                                    content = _read_plain_text(path)
                                    if not content:
                                        failures.append(f"{os.path.basename(path)}：无法读取内容")
                                        continue
                                    loaded_entries.append((os.path.basename(path), content.strip()))

                                if not loaded_entries:
                                    reason = "；".join(failures[:3]) if failures else "未能读取有效文本。"
                                    return existing_payload, f"⚠️ 上传失败：{reason}"

                                payload = dict(existing_payload or {})
                                files_entries = list(payload.get("files") or [])
                                combined_text_parts = []
                                if payload.get("combined_text"):
                                    combined_text_parts.append(payload["combined_text"])

                                for name, text in loaded_entries:
                                    files_entries.append({"name": name, "text": text, "rag_records": []})
                                    combined_text_parts.append(f"### {name}\n{text}")

                                payload["files"] = files_entries
                                payload["combined_text"] = "\n\n".join([part for part in combined_text_parts if part]).strip()
                                payload.setdefault("created_at", time.time())
                                if not payload.get("session_id"):
                                    payload["session_id"] = str(uuid4())
                                else:
                                    self.temp_chat_service.clear_session(payload["session_id"])

                                summary = f"✅ 成功载入 {len(loaded_entries)} 个文件。"
                                if failures:
                                    summary += f" 以下文件处理失败：{'; '.join(failures[:2])}"
                                return payload, summary

                            btn_chat_send.click(
                                fn=_chat_with_transcript,
                                inputs=[
                                    tb_chat_input,
                                    chat_history,
                                    state_chat_payload,
                                    slider_chat_top_k,
                                    slider_chat_similarity,
                                    state_ollama_base_url,
                                    state_ollama_model,
                                ],
                                outputs=[chat_history, tb_chat_input],
                            )
                            btn_load_chat_docs.click(
                                fn=_load_chat_documents,
                                inputs=[chat_upload, state_chat_payload],
                                outputs=[state_chat_payload, chat_upload_feedback]
                            )

                            btn_chat_clear.click(
                                fn=lambda: ([], ""),
                                inputs=None,
                                outputs=[chat_history, tb_chat_input],
                            )

                        # Translation and BGM tabs removed per request

                        with gr.TabItem("图像搜索"):
                            # Initialize face search service once
                            if not hasattr(self, "face_search"):
                                self.face_search = FaceSearchService(db_dir=os.path.join(self.args.output_dir, "face_db"))

                            with gr.Row():
                                with gr.Column(scale=4):
                                    img_query = gr.Image(
                                        type="filepath",
                                        label="上传人脸图像进行搜索",
                                        height=240,
                                        show_download_button=False,
                                    )
                                    tb_result_prefix = gr.Textbox(
                                        label="匹配结果重命名前缀（可选，仅管理员可用）",
                                        placeholder="例如：张三_",
                                        lines=1,
                                        visible=False
                                    )
                                    
                                    with gr.Accordion("搜索设置", open=False):
                                        num_top_k = gr.Slider(
                                            minimum=1,
                                            maximum=100,
                                            value=50,
                                            step=1,
                                            label="返回结果数量 (top_k)",
                                        )
                                        max_dist = gr.Slider(
                                            minimum=0.1,
                                            maximum=1.0,
                                            value=0.75,
                                            step=0.05,
                                            label="最大距离阈值（越小越相似）",
                                        )
                                    
                                    with gr.Row():
                                        btn_search = gr.Button("搜索相似图像", variant="primary")
                                        btn_rename_results = gr.Button("为搜索结果加前缀", variant="secondary", visible=False)
                                    tb_status = gr.Textbox(label="状态", interactive=False, lines=3)
                                    state_ranked_results = gr.State([])
                                    
                                    # Database Statistics
                                    with gr.Accordion("数据库统计", open=False):
                                        btn_refresh_stats = gr.Button("刷新统计", size="sm")
                                        tb_stats = gr.Textbox(
                                            label="统计信息",
                                            interactive=False,
                                            lines=4,
                                            value="点击'刷新统计'查看数据库信息。"
                                        )

                                    with gr.Accordion("数据库管理", open=False):
                                        files_to_add = gr.Files(
                                            type="filepath",
                                            label="上传图像添加到数据库（支持批量）"
                                        )
                                        btn_add = gr.Button("添加到数据库")
                                        
                                        fe_folder = gr.File(
                                            label="选择要添加的文件夹",
                                            file_count="directory",
                                            type="filepath"
                                        )
                                        cb_folder_include_sub = gr.Checkbox(
                                            label="包含子目录",
                                            value=True
                                        )
                                        btn_add_folder = gr.Button("添加文件夹到数据库")
                                        
                                        # Delete and cleanup functions
                                        files_to_delete = gr.Files(
                                            type="filepath",
                                            label="选择要从数据库删除的图像"
                                        )
                                        btn_delete = gr.Button("从数据库删除", variant="stop")
                                        
                                        btn_cleanup = gr.Button(
                                            "清理孤儿记录",
                                            size="sm"
                                        )
                                        btn_clear_db = gr.Button(
                                            "清空整个数据库",
                                            variant="stop",
                                            size="sm"
                                        )
                                        
                                with gr.Column(scale=8):
                                    gallery = gr.Gallery(
                                        label="搜索结果",
                                        columns=5,
                                        height=520,
                                        show_label=True,
                                    )

                        def _safe_rename_file(original_path: str, prefix: str):
                            base_dir = os.path.dirname(original_path)
                            basename = os.path.basename(original_path)
                            if basename.startswith(prefix):
                                return original_path, "skip"
                            name, ext = os.path.splitext(basename)
                            candidate = f"{prefix}{basename}"
                            new_path = os.path.join(base_dir, candidate)
                            counter = 1
                            while os.path.exists(new_path):
                                candidate = f"{prefix}{name}_{counter}{ext}"
                                new_path = os.path.join(base_dir, candidate)
                                counter += 1
                                if counter > 1000:
                                    return original_path, "重命名失败：达到重试上限。"
                            try:
                                os.rename(original_path, new_path)
                                updated_count, errors = self.face_search.rename_indexed_image(original_path, new_path)
                                if errors:
                                    return original_path, errors[0]
                                if updated_count == 0:
                                    logger.warning(f"未找到与 {original_path} 匹配的数据库记录，已重命名文件。")
                                return new_path, None
                            except Exception as exc:
                                logger.error(f"重命名文件失败: {original_path} -> {new_path}", exc_info=True)
                                return original_path, str(exc)

                        def _apply_prefix_to_ranked(ranked_pairs, prefix):
                            cleaned_prefix = prefix.strip()
                            if not cleaned_prefix:
                                return ranked_pairs, ""
                            renamed = 0
                            skipped = 0
                            errors = []
                            updated_pairs = []
                            for path, distance in ranked_pairs:
                                new_path, error = _safe_rename_file(path, cleaned_prefix)
                                if error:
                                    if error == "skip":
                                        skipped += 1
                                    else:
                                        errors.append(f"{os.path.basename(path)}: {error}")
                                    updated_pairs.append((path, distance))
                                else:
                                    if new_path != path:
                                        renamed += 1
                                    else:
                                        skipped += 1
                                    updated_pairs.append((new_path, distance))
                            summary_parts = []
                            if renamed:
                                summary_parts.append(f"已为 {renamed} 张图片添加前缀“{cleaned_prefix}”。")
                            if skipped:
                                summary_parts.append(f"{skipped} 张图片已包含该前缀或无需修改。")
                            if errors:
                                preview = "; ".join(errors[:3])
                                if len(errors) > 3:
                                    preview += f"... 等 {len(errors)} 项"
                                summary_parts.append(f"部分图片重命名失败：{preview}")
                            return updated_pairs, "\n".join(summary_parts).strip()

                        def _face_search(query_path: str, top_k: int, max_distance: float):
                            try:
                                if not query_path:
                                    return [], "请先上传查询图像。", []
                                
                                ranked = self.face_search.search_by_image_with_scores(
                                    query_path,
                                    top_k=int(top_k),
                                    max_distance=float(max_distance)
                                )
                                
                                if not ranked:
                                    return [], "未找到相似图像。请尝试调整搜索参数或添加更多图像到数据库。", []
                                
                                gallery_items = [
                                    (p, "")
                                    for p, _ in ranked
                                ]
                                status_msg = f"找到 {len(gallery_items)} 张相似图像。"
                                return gallery_items, status_msg, ranked
                            except ValueError as e:
                                return [], f"搜索失败: {str(e)}", []
                            except Exception as e:
                                logger.error(f"搜索出错: {e}", exc_info=True)
                                return [], f"搜索失败: {str(e)}", []

                        def _rename_ranked_results(prefix_text: str, ranked_pairs: Optional[List[Tuple[str, float]]], user_state: dict):
                            # 检查权限：只有管理员才能重命名
                            if not user_state or not user_state.get("authenticated"):
                                return [], "请先登录后再执行该操作。", ranked_pairs or []
                            if user_state.get("role") != "admin":
                                return [], "只有管理员账号才能对图片进行重命名前缀。", ranked_pairs or []
                            
                            cleaned_prefix = (prefix_text or "").strip()
                            stored_pairs = ranked_pairs or []
                            if not cleaned_prefix:
                                return [], "请输入重命名前缀后再执行该操作。", stored_pairs
                            if not stored_pairs:
                                return [], "请先执行搜索，并确保存在可重命名的结果。", stored_pairs

                            updated_pairs, summary = _apply_prefix_to_ranked(stored_pairs, cleaned_prefix)
                            gallery_items = [
                                (p, "")
                                for p, _ in updated_pairs
                            ]
                            status_msg = summary or f"为 {len(updated_pairs)} 个结果完成检查，未检测到需要重命名的文件。"
                            return gallery_items, status_msg, updated_pairs

                        def _face_add(files: list):
                            try:
                                if not files:
                                    return "请上传要添加的图像。"
                                
                                paths = [f.name if isinstance(f, gr.utils.NamedString) else f for f in files]
                                processed, faces, errors = self.face_search.add_images(paths)
                                
                                msg = f"成功索引 {processed} 张图像，添加 {faces} 个人脸。"
                                if errors:
                                    error_count = len(errors)
                                    msg += f"\n警告: {error_count} 个文件处理失败。"
                                    if error_count <= 5:
                                        msg += "\n失败详情:\n" + "\n".join(errors[:5])
                                    else:
                                        msg += f"\n前5个失败详情:\n" + "\n".join(errors[:5])
                                        msg += f"\n... 还有 {error_count - 5} 个错误"
                                
                                return msg
                            except Exception as e:
                                logger.error(f"添加图像失败: {e}", exc_info=True)
                                return f"添加失败: {str(e)}"

                        def _face_add_folder(folder_obj, include_sub: bool):
                            try:
                                from modules.face_search.service import SUPPORTED_IMAGE_EXTENSIONS

                                if not folder_obj:
                                    return "请提供有效的文件夹路径。"

                                raw_items = folder_obj if isinstance(folder_obj, (list, tuple)) else [folder_obj]
                                candidate_paths = []
                                for item in raw_items:
                                    if isinstance(item, gr.utils.NamedString):
                                        candidate_paths.append(item.name)
                                    elif item:
                                        try:
                                            candidate_paths.append(os.fspath(item))
                                        except TypeError:
                                            continue
                                if not candidate_paths:
                                    return "请提供有效的文件夹路径。"

                                files_to_add = set()
                                visited_dirs = set()

                                def collect_from_directory(dir_path: str):
                                    if not dir_path or dir_path in visited_dirs:
                                        return
                                    visited_dirs.add(dir_path)
                                    if include_sub:
                                        for root, _, files in os.walk(dir_path):
                                            for fn in files:
                                                if os.path.splitext(fn)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS:
                                                    files_to_add.add(os.path.join(root, fn))
                                    else:
                                        for fn in os.listdir(dir_path):
                                            fp = os.path.join(dir_path, fn)
                                            if os.path.isfile(fp) and os.path.splitext(fn)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS:
                                                files_to_add.add(fp)

                                for path in candidate_paths:
                                    if not path:
                                        continue
                                    if os.path.isdir(path):
                                        collect_from_directory(path)
                                    elif os.path.isfile(path):
                                        if os.path.splitext(path)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS:
                                            files_to_add.add(path)

                                if not files_to_add:
                                    return "文件夹中未找到支持的图像文件。"

                                processed, faces, errors = self.face_search.add_images(files_to_add)

                                msg = f"成功索引 {processed} 张图像，添加 {faces} 个人脸。"
                                if errors:
                                    error_count = len(errors)
                                    msg += f"\n警告: {error_count} 个文件处理失败。"
                                    if error_count <= 5:
                                        msg += "\n失败详情:\n" + "\n".join(errors[:5])
                                    else:
                                        msg += f"\n前5个失败详情:\n" + "\n".join(errors[:5])
                                        msg += f"\n... 还有 {error_count - 5} 个错误"

                                return msg
                            except Exception as e:
                                logger.error(f"添加文件夹失败: {e}", exc_info=True)
                                return f"添加文件夹失败: {str(e)}"
                        
                        def _face_delete(files: list):
                            try:
                                if not files:
                                    return "请选择要删除的图像。"
                                
                                paths = [f.name if isinstance(f, gr.utils.NamedString) else f for f in files]
                                deleted_count, errors = self.face_search.delete_images(paths)
                                
                                msg = f"成功删除 {deleted_count} 个人脸记录。"
                                if errors:
                                    msg += f"\n警告: {len(errors)} 个错误。\n" + "\n".join(errors[:3])
                                
                                return msg
                            except Exception as e:
                                logger.error(f"删除失败: {e}", exc_info=True)
                                return f"删除失败: {str(e)}"
                        
                        def _refresh_stats():
                            try:
                                stats = self.face_search.get_statistics()
                                msg = f"""数据库统计信息:
总人脸数: {stats['total_faces']}
唯一图像数: {stats['total_images']}
已索引文件数: {stats['total_indexed_files']}"""
                                return msg
                            except Exception as e:
                                logger.error(f"获取统计信息失败: {e}", exc_info=True)
                                return f"获取统计信息失败: {str(e)}"
                        
                        def _cleanup_orphaned():
                            try:
                                deleted_count, errors = self.face_search.remove_orphaned_entries()
                                msg = f"清理完成: 删除了 {deleted_count} 个孤儿记录。"
                                if errors:
                                    msg += f"\n警告: {len(errors)} 个错误。"
                                return msg
                            except Exception as e:
                                logger.error(f"清理失败: {e}", exc_info=True)
                                return f"清理失败: {str(e)}"
                        
                        def _clear_database():
                            try:
                                success = self.face_search.clear_database()
                                if success:
                                    return "数据库已清空。"
                                else:
                                    return "清空数据库失败，请查看日志。"
                            except Exception as e:
                                logger.error(f"清空数据库失败: {e}", exc_info=True)
                                return f"清空数据库失败: {str(e)}"

                        btn_search.click(
                            fn=_face_search,
                            inputs=[img_query, num_top_k, max_dist],
                            outputs=[gallery, tb_status, state_ranked_results]
                        )
                        btn_rename_results.click(
                            fn=_rename_ranked_results,
                            inputs=[tb_result_prefix, state_ranked_results, auth_state],
                            outputs=[gallery, tb_status, state_ranked_results]
                        )
                        btn_add.click(fn=_face_add, inputs=[files_to_add], outputs=[tb_status])
                        btn_add_folder.click(
                            fn=_face_add_folder,
                            inputs=[fe_folder, cb_folder_include_sub],
                            outputs=[tb_status]
                        )
                        btn_delete.click(
                            fn=_face_delete,
                            inputs=[files_to_delete],
                            outputs=[tb_status]
                        )
                        btn_refresh_stats.click(
                            fn=_refresh_stats,
                            outputs=[tb_stats]
                        )
                        btn_cleanup.click(
                            fn=_cleanup_orphaned,
                            outputs=[tb_status]
                        )
                        btn_clear_db.click(
                            fn=_clear_database,
                            outputs=[tb_status]
                        )

            def _handle_register(username, password):
                success, message = self.register_user(username, password)
                prefix = "✅" if success else "⚠️"
                return f"{prefix} {message}"

            def _handle_login(username, password, current_state):
                success, role, message = self.login_user(username, password)
                dropdown_update = gr.update(value=None, choices=[])
                user_role_display = gr.update(value="")
                is_admin = False
                is_main_admin = False
                if success:
                    normalized_username = (username or "").strip()
                    new_state = {
                        "username": normalized_username,
                        "role": role,
                        "authenticated": True
                    }
                    is_admin = (role == "admin")
                    is_main_admin = (normalized_username == self.default_admin_username)
                    if is_admin:
                        pending = self.get_pending_users()
                        dropdown_update = gr.update(
                            choices=pending,
                            value=pending[0] if pending else None
                        )
                    role_label = "管理员" if role == "admin" else "普通用户"
                    user_role_display = gr.update(
                        value=f"当前登录账号：**{normalized_username}**（{role_label}）"
                    )
                    return (
                        new_state,
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=is_admin),
                        gr.update(visible=is_main_admin),
                        gr.update(visible=is_admin),
                        f"✅ {message}",
                        dropdown_update,
                        user_role_display,
                        gr.update(visible=is_admin),
                        gr.update(visible=is_admin)
                    )

                fallback_state = _empty_user_state()
                return (
                    fallback_state,
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    f"⚠️ {message}",
                    dropdown_update,
                    user_role_display,
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

            def _handle_logout(current_state):
                message = "当前未登录。"
                if current_state and current_state.get("authenticated"):
                    message = f"用户 {current_state.get('username')} 已退出登录。"
                return (
                    _empty_user_state(),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    message,
                    gr.update(value=None, choices=[]),
                    gr.update(value=""),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

            def _refresh_pending(user_state):
                if not user_state or not user_state.get("authenticated"):
                    return gr.update(), "请先登录后再操作。"
                if user_state.get("role") != "admin":
                    return gr.update(), "仅管理员可以审核用户。"
                pending = self.get_pending_users()
                if pending:
                    return gr.update(choices=pending, value=pending[0]), f"共有 {len(pending)} 个待审核用户。"
                return gr.update(choices=[], value=None), "暂无待审核用户。"

            def _approve_pending(selected_user, user_state):
                if not user_state or not user_state.get("authenticated"):
                    return gr.update(), "请先登录后再操作。"
                if user_state.get("role") != "admin":
                    return gr.update(), "仅管理员可以审核用户。"
                success, message = self.approve_user(selected_user)
                pending = self.get_pending_users()
                dropdown_update = gr.update(choices=pending, value=pending[0] if pending else None)
                prefix = "✅" if success else "⚠️"
                return dropdown_update, f"{prefix} {message}"

            def _refresh_users_list(user_state):
                if not user_state or not user_state.get("authenticated"):
                    return gr.update(), "", "请先登录后再操作。"
                if user_state.get("username") != self.default_admin_username:
                    return gr.update(), "", "仅主账号可以管理用户权限。"
                users = self.get_all_users()
                if not users:
                    return gr.update(choices=[], value=None), "", "暂无其他用户。"
                choices = [f"{u['username']} ({'管理员' if u['role'] == 'admin' else '普通用户'})" for u in users]
                selected_user = users[0] if users else None
                selected_display = f"{selected_user['username']} ({'管理员' if selected_user['role'] == 'admin' else '普通用户'})" if selected_user else None
                user_info = f"**当前用户列表：**\n" + "\n".join([f"- {u['username']}: {u['role']} ({u['status']})" for u in users])
                return gr.update(choices=choices, value=selected_display), user_info, ""

            def _on_user_selected(selected_value, user_state):
                if not selected_value or not user_state or not user_state.get("authenticated"):
                    return ""
                if user_state.get("username") != self.default_admin_username:
                    return ""
                # 从选择的值中提取用户名（格式：username (role)）
                username = selected_value.split(" (")[0] if " (" in selected_value else selected_value
                users = self.get_all_users()
                for u in users:
                    if u['username'] == username:
                        return f"**选中用户：** {u['username']}\n**当前角色：** {u['role']}\n**账号状态：** {u['status']}"
                return ""

            def _grant_admin(selected_user, user_state):
                if not user_state or not user_state.get("authenticated"):
                    return gr.update(), "", "请先登录后再操作。"
                if user_state.get("username") != self.default_admin_username:
                    return gr.update(), "", "仅主账号可以赋予管理员权限。"
                if not selected_user:
                    return gr.update(), "", "请选择要赋予管理员权限的用户。"
                # 从选择的值中提取用户名
                username = selected_user.split(" (")[0] if " (" in selected_user else selected_user
                success, message = self.grant_admin_role(username, user_state.get("username"))
                # 刷新用户列表
                users = self.get_all_users()
                choices = [f"{u['username']} ({'管理员' if u['role'] == 'admin' else '普通用户'})" for u in users]
                user_info = f"**当前用户列表：**\n" + "\n".join([f"- {u['username']}: {u['role']} ({u['status']})" for u in users])
                prefix = "✅" if success else "⚠️"
                return gr.update(choices=choices, value=selected_user if selected_user in choices else (choices[0] if choices else None)), user_info, f"{prefix} {message}"

            def _rebuild_rag_index(user_state):
                if not self.text_corrector:
                    return "⚠️ RAG 文本纠错未启用，无法更新知识库索引。"
                if not user_state or not user_state.get("authenticated"):
                    return "请先登录后再操作。"
                if user_state.get("role") != "admin":
                    return "仅管理员可以更新知识库索引。"
                kb_dir = self.args.rag_kb_dir
                try:
                    gr.Info("开始重建知识库索引，请稍候...")
                    message = self.text_corrector.rebuild_index(kb_dir)
                    logger.info("RAG 知识库索引已更新：%s", message)
                    return f"✅ {message}"
                except Exception as exc:
                    logger.error("更新 RAG 知识库索引失败：%s", exc, exc_info=True)
                    return f"⚠️ 更新失败：{exc}"

            def _revoke_admin(selected_user, user_state):
                if not user_state or not user_state.get("authenticated"):
                    return gr.update(), "", "请先登录后再操作。"
                if user_state.get("username") != self.default_admin_username:
                    return gr.update(), "", "仅主账号可以撤销管理员权限。"
                if not selected_user:
                    return gr.update(), "", "请选择要撤销管理员权限的用户。"
                # 从选择的值中提取用户名
                username = selected_user.split(" (")[0] if " (" in selected_user else selected_user
                success, message = self.revoke_admin_role(username, user_state.get("username"))
                # 刷新用户列表
                users = self.get_all_users()
                choices = [f"{u['username']} ({'管理员' if u['role'] == 'admin' else '普通用户'})" for u in users]
                user_info = f"**当前用户列表：**\n" + "\n".join([f"- {u['username']}: {u['role']} ({u['status']})" for u in users])
                prefix = "✅" if success else "⚠️"
                return gr.update(choices=choices, value=selected_user if selected_user in choices else (choices[0] if choices else None)), user_info, f"{prefix} {message}"

            btn_register.click(
                fn=_handle_register,
                inputs=[tb_login_username, tb_login_password],
                outputs=[login_feedback]
            )
            btn_login.click(
                fn=_handle_login,
                inputs=[tb_login_username, tb_login_password, auth_state],
                outputs=[auth_state, login_view, main_view, admin_panel, admin_role_panel, rag_panel, login_feedback, pending_users_dropdown, user_info_box, tb_result_prefix, btn_rename_results]
            )
            btn_logout.click(
                fn=_handle_logout,
                inputs=[auth_state],
                outputs=[auth_state, login_view, main_view, admin_panel, admin_role_panel, rag_panel, login_feedback, pending_users_dropdown, user_info_box, tb_result_prefix, btn_rename_results]
            )
            btn_refresh_pending.click(
                fn=_refresh_pending,
                inputs=[auth_state],
                outputs=[pending_users_dropdown, admin_feedback]
            )
            btn_approve_pending.click(
                fn=_approve_pending,
                inputs=[pending_users_dropdown, auth_state],
                outputs=[pending_users_dropdown, admin_feedback]
            )
            btn_refresh_users.click(
                fn=_refresh_users_list,
                inputs=[auth_state],
                outputs=[all_users_dropdown, user_role_display, admin_role_feedback]
            )
            all_users_dropdown.change(
                fn=_on_user_selected,
                inputs=[all_users_dropdown, auth_state],
                outputs=[user_role_display]
            )
            btn_grant_admin.click(
                fn=_grant_admin,
                inputs=[all_users_dropdown, auth_state],
                outputs=[all_users_dropdown, user_role_display, admin_role_feedback]
            )
            btn_revoke_admin.click(
                fn=_revoke_admin,
                inputs=[all_users_dropdown, auth_state],
                outputs=[all_users_dropdown, user_role_display, admin_role_feedback]
            )
            btn_update_kb.click(
                fn=_rebuild_rag_index,
                inputs=[auth_state],
                outputs=[rag_update_feedback]
            )

        # Launch the app with optional gradio settings
        args = self.args
        self.app.queue(
            api_open=args.api_open
        ).launch(
            share=args.share,
            server_name=args.server_name,
            server_port=args.server_port,
            root_path=args.root_path if args.root_path else None,
            inbrowser=args.inbrowser,
            ssl_verify=args.ssl_verify,
            ssl_keyfile=args.ssl_keyfile,
            ssl_keyfile_password=args.ssl_keyfile_password,
            ssl_certfile=args.ssl_certfile,
            allowed_paths=eval(args.allowed_paths) if args.allowed_paths else None
        )

    @staticmethod
    def open_folder(folder_path: str):
        if os.path.exists(folder_path):
            os.system(f"start {folder_path}")
        else:
            os.makedirs(folder_path, exist_ok=True)
            logger.info(f"The directory path {folder_path} has newly created.")


parser = argparse.ArgumentParser()
parser.add_argument('--whisper_type', type=str, default=WhisperImpl.FASTER_WHISPER.value,
                    choices=[item.value for item in WhisperImpl],
                    help='A type of the whisper implementation (Github repo name)')
parser.add_argument('--share', type=str2bool, default=False, nargs='?', const=True, help='Gradio share value')
parser.add_argument('--server_name', type=str, default='0.0.0.0', help='Gradio server host')
parser.add_argument('--server_port', type=int, default=None, help='Gradio server port')
parser.add_argument('--root_path', type=str, default=None, help='Gradio root path')
parser.add_argument('--username', type=str, default=None, help='Gradio authentication username')
parser.add_argument('--password', type=str, default=None, help='Gradio authentication password')
parser.add_argument('--theme', type=str, default=None, help='Gradio Blocks theme')
parser.add_argument('--colab', type=str2bool, default=False, nargs='?', const=True, help='Is colab user or not')
parser.add_argument('--api_open', type=str2bool, default=False, nargs='?', const=True,
                    help='Enable api or not in Gradio')
parser.add_argument('--allowed_paths', type=str, default=None, help='Gradio allowed paths')
parser.add_argument('--inbrowser', type=str2bool, default=True, nargs='?', const=True,
                    help='Whether to automatically start Gradio app or not')
parser.add_argument('--ssl_verify', type=str2bool, default=True, nargs='?', const=True,
                    help='Whether to verify SSL or not')
parser.add_argument('--auth_db_path', type=str, default=DEFAULT_AUTH_DB_PATH, help='Path to the SQLite database used for UI login')
parser.add_argument('--default_admin_username', type=str, default=DEFAULT_ADMIN_USERNAME,
                    help='Bootstrap admin username if database is empty')
parser.add_argument('--default_admin_password', type=str, default=DEFAULT_ADMIN_PASSWORD,
                    help='Bootstrap admin password if database is empty')
parser.add_argument('--ssl_keyfile', type=str, default=None, help='SSL Key file location')
parser.add_argument('--ssl_keyfile_password', type=str, default=None, help='SSL Key file password')
parser.add_argument('--ssl_certfile', type=str, default=None, help='SSL cert file location')
parser.add_argument('--whisper_model_dir', type=str, default=WHISPER_MODELS_DIR,
                    help='Directory path of the whisper model')
parser.add_argument('--faster_whisper_model_dir', type=str, default=FASTER_WHISPER_MODELS_DIR,
                    help='Directory path of the faster-whisper model')
parser.add_argument('--insanely_fast_whisper_model_dir', type=str,
                    default=INSANELY_FAST_WHISPER_MODELS_DIR,
                    help='Directory path of the insanely-fast-whisper model')
parser.add_argument('--diarization_model_dir', type=str, default=DIARIZATION_MODELS_DIR,
                    help='Directory path of the diarization model')
parser.add_argument('--nllb_model_dir', type=str, default=NLLB_MODELS_DIR,
                    help='Directory path of the Facebook NLLB model')
parser.add_argument('--uvr_model_dir', type=str, default=UVR_MODELS_DIR,
                    help='Directory path of the UVR model')
parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory path of the outputs')
parser.add_argument('--rag_kb_dir', type=str, default=KNOWLEDGE_BASE_DIR,
                    help='Directory path of txt knowledge base used for RAG correction')
parser.add_argument('--rag_store_dir', type=str, default=RAG_STORE_DIR,
                    help='Persistent directory for RAG vector store')
parser.add_argument('--rag_embedding_model', type=str,
                    default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    help='SentenceTransformer model name for RAG correction')
_args = parser.parse_args()

if __name__ == "__main__":
    
    
    app = App(args=_args)
    app.launch()
    
