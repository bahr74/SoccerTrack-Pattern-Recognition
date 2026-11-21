"""Presentation of saved Member 1 artifacts; never fabricates missing results."""
from pathlib import Path
import scientific_workflow as workflow
from experiment import OUT

def dashboard():
    prior=OUT/'experiment_provisional.json'
    result={'status':workflow.STATUS,'prior_experiment_available':prior.is_file(),'expanded_available':False}
    try: run=workflow.active_run()
    except FileNotFoundError: return result
    result.update(expanded_available=True,manifest=workflow.read_json(run/'manifest.json'))
    result['artifacts']=[{'name':p.name,'file':p.relative_to(workflow.BASE).as_posix(),'group':p.parent.name}
        for p in sorted(run.rglob('*')) if p.is_file() and p.suffix.lower() in ('.png','.json','.csv','.mp4')]
    pointer=workflow.BASE/'corrections/latest.json'
    if pointer.is_file():
        correction=workflow.read_json(pointer)
        if correction.get('source_run')==workflow.rel(run):
            folder=workflow.ROOT/correction['path'];supp=workflow.read_json(folder/'historical_audit_supplement.json')
            result['audit_supplement']=supp
            for original,override in supp['presentation_override'].items():
                for artifact in result['artifacts']:
                    if artifact['file']==(run/original).relative_to(workflow.BASE).as_posix():artifact['file']=(workflow.ROOT/override).relative_to(workflow.BASE).as_posix()
            result['artifacts'].extend({'name':p.name,'file':p.relative_to(workflow.BASE).as_posix(),'group':'corrective_audit'} for p in folder.glob('*.json'))
    return result
