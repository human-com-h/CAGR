from pathlib import Path
import io,sys,unittest,zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from check_publication_scope import inspect_payload

class PublicationScope(unittest.TestCase):
    def test_reject_paper_formats(self):
        for name in ['main.pdf','main.tex','references.bib','draft.docx','submission/manuscript.md']:
            self.assertTrue(inspect_payload(name,b'not public'))
    def test_reject_renamed_full_documents(self):
        self.assertTrue(inspect_payload('notes.bin',b'%PDF-1.7 content'))
        tex=b'\\documentclass{article}\n'+b'\\begin{document}\nbody'
        self.assertTrue(inspect_payload('notes.txt',tex))
        text='\n'.join(['# Introduction','body','# Conclusion','body','# References','body']).encode()
        self.assertTrue(inspect_payload('notes.md',text))
    def test_reject_nested_archive_documents(self):
        inside=io.BytesIO()
        with zipfile.ZipFile(inside,'w') as z:z.writestr('main.tex','internal')
        outside=io.BytesIO()
        with zipfile.ZipFile(outside,'w') as z:z.writestr('sources.zip',inside.getvalue())
        self.assertTrue(inspect_payload('release.zip',outside.getvalue()))
    def test_allow_metadata_and_table_generator(self):
        self.assertFalse(inspect_payload('README.md',b'# CAGR\n## Abstract\nOverview\n## Citation\nCitation data'))
        self.assertFalse(inspect_payload('tools/table.py',b"table = r'\\begin{table}'"))
        self.assertFalse(inspect_payload('evidence/metrics.csv',b'method,em\na,0.5\n'))

if __name__=='__main__':unittest.main()
