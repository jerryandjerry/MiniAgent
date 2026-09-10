"""Administrative-place-name normalization shared by tools and RAG."""

ADMIN_SUFFIXES = (
    "土家族", "苗族", "藏族", "彝族", "白族", "傣族", "景颇族", "朝鲜族",
    "哈萨克", "柯尔克孜", "蒙古", "回族", "壮族", "维吾尔", "哈尼族",
    "布依族", "侗族", "羌族", "傈僳族", "自治州", "自治县", "自治区",
    "地区", "盟", "市", "省", "县",
)


def stem(name: str) -> str:
    """Return the stable geographic stem used for guide-name matching."""
    out = (name or "").strip()
    for suffix in ADMIN_SUFFIXES:
        index = out.find(suffix)
        if index > 1:
            out = out[:index]
    return out.strip()

