# 延迟翻译，避免在模块导入时调用 gettext（需要 Gradio 请求上下文）
# 使用一个简单的类来包装，在需要时再获取翻译
class _AutomaticDetection:
    """延迟翻译的自动检测选项"""
    def __init__(self):
        self._key = "Automatic Detection"
    
    def unwrap(self):
        """返回原始字符串，用于比较"""
        return self._key
    
    def __str__(self):
        """返回翻译后的字符串（在 Gradio 上下文中）"""
        try:
            from gradio_i18n import gettext as _
            return str(_(self._key))
        except (LookupError, AttributeError, ImportError, ModuleNotFoundError):
            # 如果没有 Gradio 上下文或模块未安装，返回原始字符串
            return self._key
    
    def __repr__(self):
        return f"<AutomaticDetection: {self._key}>"
    
    def __eq__(self, other):
        """支持与字符串比较"""
        if isinstance(other, str):
            return self._key == other
        return self is other
    
    def __hash__(self):
        """使对象可哈希，以便用作字典键"""
        return hash(self._key)

AUTOMATIC_DETECTION = _AutomaticDetection()
GRADIO_NONE_STR = ""
GRADIO_NONE_NUMBER_MAX = 9999
GRADIO_NONE_NUMBER_MIN = 0
