import re


""" Rewrites and expands a query in a single step. Combines: rule-based rewriting, synonym expansion, domain enrichment ----------- """
def rewrite_and_expand_query(query: str, synonym_dict: dict = None):

    original_query = query
    query_lower = query.lower()
    original_words = re.findall(r'\b\w+\b', query_lower)     # Checks for original words
    expanded_terms = set()
    """ expanded_terms = rule_based_core(query_lower) # Check for technical words """
    expanded_terms.update(rule_based_core(query_lower))
    
    # SYNONYM EXPANSION
    if synonym_dict:
        for word in original_words:
            if word in synonym_dict:
                expanded_terms.update(synonym_dict[word])

    # Delete duplicated words
    expanded_terms = {
        term for term in expanded_terms
        if term.lower() not in original_words
    }

    # Set size limit
    MAX_TERMS = 15
    expanded_terms = list(expanded_terms)[:MAX_TERMS]

    # FINAL QUERY
    final_query = original_query + " " + " ".join(expanded_terms)

    return final_query.strip()


""" Domain-specific rule-based query rewriting using extracted synonym groups -------------------------------------------------------------"""
def rule_based_core(query: str):

    query = query.lower()
    expanded_terms = set()
    if any(w in query for w in ["sm6-36", "sm6 36", "sm6-24", "sm6 24", "premset", "cbgs", "airset"]):
        expanded_terms.update([
            "familia"
        ])

    # SUPPLIER / PROVEEDOR
    if any(w in query for w in ["proveedor", "supplier", "vendor", "proveedores", "suppliers"]):
        expanded_terms.update([
            "proveedor", "vendedor"
        ])

    # PERFORMANCE
    if any(w in query for w in ["indicador", "indicadores", "performance", "desempeño"]):
        expanded_terms.update([
            "KPI", "medir", "evaluación", "performance"
        ])

    # DEFECTOS / PROBLEMAS / NO CONFORMIDAD
    if any(w in query for w in ["defecto", "defectos", "problema", "problemas", "issue", "nc", "no conformidad", "no conforme"]):
        expanded_terms.update([
            "defecto", "problema", "no conformidad", "no conforme"
        ])

    # CUARENTENA
    """ if any(w in query for w in ["cuarentena", "zona no conforme"]):
        expanded_terms.update([
            "quarantine", "segregation",
            "non-conforming material",
            "hold area"
        ]) """

    # TRAZABILIDAD
    """ if any(w in query for w in ["trazabilidad", "trazables"]):
        expanded_terms.update([
            "traceability", "serial tracking",
            "identification", "tracking"
        ]) """

    # CELDA / EQUIPO
    """ if any(w in query for w in ["celda", "celdas", "modelo", "IM", "IMB", "IME" ,"QM", "QMB" ,"QME" ,"GAM" ,"GBM", "CM"]):
        expanded_terms.update([
            "cell", "equipment",
            "assembly", "cubicle"
        ]) """


    # CONTROL
    if any(w in query for w in ["control", "controles"]):
        expanded_terms.update([
            "control", "control de proceso",
            "control de calidad"
        ])

    # SEGURIDAD
    """ if "seguridad" in query:
        expanded_terms.update([
            "safety", "risk",
            "safety requirement"
        ])
 """
    return expanded_terms
