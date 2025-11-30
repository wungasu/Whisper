from typing import List
from modules.whisper.data_classes import Segment

_cc = None
_use_opencc = False
_use_zhconv = False

# Try to initialize converters once
try:
    from opencc import OpenCC  # type: ignore
    _cc = OpenCC('t2s')
    _use_opencc = True
except Exception:
    try:
        from zhconv import convert as _zh_convert  # type: ignore
        _use_zhconv = True
    except Exception:
        pass


def t2s(text: str) -> str:
    if not text:
        return text
    if _use_opencc and _cc is not None:
        return _cc.convert(text)
    if _use_zhconv:
        return _zh_convert(text, 'zh-cn')
    return text


def convert_segments_to_simplified(segments: List[Segment]) -> List[Segment]:
    if not segments:
        return segments
    converted: List[Segment] = []
    for seg in segments:
        new_seg = seg.model_copy(deep=True)
        if new_seg.text:
            new_seg.text = t2s(new_seg.text)
        if new_seg.words:
            for w in new_seg.words:
                if w.word:
                    w.word = t2s(w.word)
        converted.append(new_seg)
    return converted


