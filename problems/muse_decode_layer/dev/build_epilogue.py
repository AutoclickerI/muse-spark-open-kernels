from pathlib import Path
p=Path(__file__).resolve().parents[1]
s=(p/'cand_chain6.py').read_text().split('# Persistent dense islands:')[0]
s+='\n\n'+(p/'dev/epilogue_runtime.py').read_text()
(p/'cand_epilogue.py').write_text(s)
