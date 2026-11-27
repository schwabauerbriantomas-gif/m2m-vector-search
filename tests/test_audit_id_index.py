"""[cierre] El índice de hallazgos del CHANGELOG no miente.

El CHANGELOG dice que cada id ``[audit …]`` (y cada etiqueta ``[P…]`` de un pack) o nombra
un test bajo ``tests/`` que lo cita, o lleva ``no unit test`` con el motivo
(documentación, CI, packaging, despliegue). Este test lo mantiene verdadero: agrega un id
al CHANGELOG y vas a tener que citarlo desde un test o declarar la excepción.

Incluye dos afirmaciones del CHANGELOG que no tenían test propio:
``enable_lsh_fallback`` apagado por defecto [audit r3a-14] y ``test_lsh_speedup``
reemplazado por el test de paridad ``test_lsh_matches_exact_scan`` [audit r3b-08].
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BRACKET = re.compile(r"\[(audit )?([^\]]+)\]")
ID = re.compile(r"[A-Za-z]*\d+[a-z]?(?:-\d+)?")


def _changelog_ids():
    """id -> True si su corchete declara `no unit test`."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    out = {}
    for m in BRACKET.finditer(text):
        content = m.group(2)
        if not (m.group(1) or re.match(r"P\d", content)):
            continue
        content = re.sub(r"^audit\s+", "", content)
        exempt = "no unit test" in content
        for tok in re.split(r"[,;]", content):
            tok = tok.strip().strip("`")
            if ID.fullmatch(tok):
                out[tok] = out.get(tok, False) or exempt
    return out


def _tests_text():
    return "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "tests").glob("*.py"))


def test_changelog_tags_index_is_not_empty():
    ids = _changelog_ids()
    assert len(ids) >= 60, len(ids)


@pytest.mark.parametrize("audit_id", sorted(_changelog_ids()))
def test_each_changelog_id_has_a_citing_test_or_says_so(audit_id):
    ids = _changelog_ids()
    pat = re.compile(r"(?<![A-Za-z0-9_-])" + re.escape(audit_id) + r"(?![A-Za-z0-9_-])")
    if pat.search(_tests_text()):
        return
    assert ids[audit_id], (
        f"`{audit_id}` del CHANGELOG no aparece en ningún tests/*.py: citarlo desde el test "
        "que lo cubre o agregar `no unit test: <motivo>` en su corchete"
    )


def test_lsh_fallback_is_off_by_default():
    """[audit r3a-14] el índice LSH es opt-in."""
    from m2m import SimpleVectorDB

    assert SimpleVectorDB(latent_dim=8).enable_lsh_fallback is False


def test_lsh_parity_test_replaced_the_speedup_test():
    """[audit r3b-08] ``test_lsh_speedup`` pasó a ser el test de paridad."""
    text = (ROOT / "tests" / "test_lsh.py").read_text(encoding="utf-8")
    assert "def test_lsh_matches_exact_scan(" in text
    assert "def test_lsh_speedup(" not in text
