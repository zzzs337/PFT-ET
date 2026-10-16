"""Source-package selection; runtime products remain in their run directories."""
OMIT_NAMES={
    'DELIVERY_ZH.md','PACKAGE_AUDIT_REPORT_ZH.md','PACKAGE_GUIDE_ZH.md',
    'REVIEWER_EVIDENCE_MAP.csv','RELEASE_CHECKLIST.md','SOURCE_PROVENANCE.json',
    'IMPLEMENTATION_AND_RELEASE_NOTES.md','EVIDENCE_INVENTORY.md','verification.md',
    'CHANGELOG_2.2.md','VERIFICATION_2.2.md',
}


def selected(root):
    for path in sorted(root.rglob('*')):
        if not path.is_file():continue
        rel=path.relative_to(root)
        if rel.parts[:2]==('data','reported_results'):continue
        if rel.parts[:2]==('examples','generated'):continue
        if path.name in OMIT_NAMES or any((p.startswith('.') and p!='.gitignore') or p in ('__pycache__','outputs') for p in rel.parts):continue
        if rel.parts[:2]==('docs','verification') and (len(rel.parts)<3 or rel.parts[2] != '2.3.0'):continue
        yield path
