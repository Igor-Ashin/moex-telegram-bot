# Секторы и акции (перенесено из main.py)

SECTORS = {
    "Финансы": ["SBER", "T", "VTBR", "MOEX", "SPBE", "RENI", "BSPB", "SVCB", "MBNK", "LEAS", "SFIN", "AFKS", "CARM", "ZAYM", "MGKL"],
    "Нефтегаз": ["GAZP", "NVTK", "LKOH", "ROSN", "TATNP", "TATN", "SNGS", "SNGSP", "BANE", "BANEP", "RNFT"],
    "Металлы и добыча": ["ALRS", "GMKN", "RUAL", "TRMK", "MAGN", "NLMK", "CHMF", "MTLRP", "MTLR", "RASP", "PLZL", "UGLD", "SGZH"],
    "IT": ["YDEX", "DATA", "HEAD", "POSI", "VKCO", "ASTR", "IVAT", "DELI", "WUSH", "CNRU", "DIAS", "SOFL", "ELMT"],
    "Телеком": ["MTSS", "RTKMP", "RTKM", "MGTSP"],
    "Строители": ["SMLT", "PIKK", "LSRG"],
    "Ритейл": ["X5", "MGNT", "LENT", "BELU", "OZON", "EUTR", "ABRD", "GCHE", "AQUA", "HNFG", "MVID", "VSEH"],
    "Электро": ["IRAO", "UPRO", "LSNGP", "MSRS", "MRKU", "MRKC", "MRKP", "FEES", "HYDR", "ELFV"],
    "Транспорт и логистика": ["TRNFP", "AFLT", "FESH", "NMTP", "FLOT"],
    "Агро": ["PHOR", "RAGR", "KZOS", "AKRN", "NKHP"],
    "Медицина": ["MDMG", "OZPH", "PRMD", "ABIO", "GEMC"],
    "Машиностроение": ["UWGN", "SVAV", "KMAZ", "UNAC", "IRKT"]
}

# Упрощенная версия для некоторых функций
SECTORS_SIMPLIFIED = {
    "Финансы": ["SBER", "T", "VTBR", "MOEX", "SPBE", "RENI", "BSPB", "SVCB", "MBNK", "LEAS", "SFIN", "AFKS"],
    "Нефтегаз": ["GAZP", "NVTK", "LKOH", "ROSN", "TATNP", "TATN", "SNGS", "SNGSP", "BANE", "BANEP", "RNFT"],
    "Металлы и добыча": ["ALRS", "GMKN", "RUAL", "TRMK", "MAGN", "NLMK", "CHMF", "MTLRP", "MTLR", "PLZL", "SGZH"],
    "IT": ["YDEX", "DATA", "HEAD", "POSI", "VKCO", "ASTR", "DELI", "WUSH", "CNRU", "DIAS"],
    "Телеком": ["MTSS", "RTKMP", "RTKM"],
    "Строители": ["SMLT", "PIKK"],
    "Ритейл": ["X5", "MGNT", "LENT", "BELU", "OZON", "EUTR", "ABRD", "GCHE", "AQUA", "HNFG", "MVID"],
    "Электро": ["IRAO", "UPRO", "LSNGP", "MRKP"],
    "Транспорт и логистика": ["TRNFP", "AFLT", "FESH", "NMTP", "FLOT"],
    "Агро": ["PHOR", "RAGR"],
    "Медицина": ["MDMG", "OZPH", "PRMD"],
    "Машиностроение": ["UWGN", "SVAV"]
}

def get_all_tickers() -> list[str]:
    """Возвращает все тикеры из всех секторов"""
    return sum(SECTORS.values(), [])

def get_sector_tickers(sector: str) -> list[str]:
    """Возвращает тикеры для конкретного сектора"""
    return SECTORS.get(sector, [])

def get_simplified_tickers() -> list[str]:
    """Возвращает упрощенный список тикеров"""
    return sum(SECTORS_SIMPLIFIED.values(), [])