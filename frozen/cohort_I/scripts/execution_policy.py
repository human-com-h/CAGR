"""Enforce project submission scope and bounded chat waiting policy."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

class PolicyError(RuntimeError):
    pass

def frozen_runtime(policy,root=ROOT):
    return policy.get('execution_context')=='frozen_kaggle_runtime' and root.resolve().is_relative_to(Path('/kaggle/working').resolve()) and (root/'package_manifest.json').is_file()

def load_policy(root=ROOT):
    path = root / 'configs/execution_policy.yaml'
    if not path.is_file():
        raise PolicyError('Missing configs/execution_policy.yaml; execution is blocked.')
    policy = yaml.safe_load(path.read_text(encoding='utf-8'))
    submission=policy['experiment_submission']
    if submission['mode'] not in {'user_manual','agent_authorized_once','agent_submit_then_handoff','local_only'}:
        raise PolicyError('Unknown execution mode; submission is blocked.')
    if submission['agent_may_restart_cloud_run']:
        raise PolicyError('Automatic cloud restart remains forbidden.')
    if submission['mode']=='user_manual' and submission['agent_may_submit']:
        raise PolicyError('Manual policy must forbid agent submission.')
    if submission['mode'] in {'agent_authorized_once','agent_submit_then_handoff'} and (not submission['agent_may_submit'] or not submission.get('authorization')):
        raise PolicyError('Agent submission requires recorded direct user authorization.')
    if submission['mode']=='local_only' and (submission['agent_may_submit'] or policy['assets'].get('agent_may_prepare_and_upload_private_assets')):
        raise PolicyError('Local-only work must forbid cloud submission and asset upload.')
    if submission['mode']=='agent_submit_then_handoff':
        wait=policy.get('chat_wait',{})
        maximum=wait.get('max_wait_seconds',0)
        interval=wait.get('poll_interval_seconds',0)
        request_timeout=wait.get('status_request_timeout_seconds',0)
        if not (0 < maximum <= 300 and 0 < interval <= 60 and 0 < request_timeout <= 60):
            raise PolicyError('Chat polling requires finite waits: total <=300s, interval/request <=60s.')
        if (wait.get('on_timeout')!='save_run_state_and_end_turn'
                or wait.get('on_status_error')!='save_last_state_and_end_turn'
                or wait.get('remote_run_on_chat_end')!='continue'
                or wait.get('automatic_background_polling') is not False
                or wait.get('automatic_thread_resume') is not False):
            raise PolicyError('Long cloud runs must be handed off without cancellation or background polling.')
    return policy

def guard_cloud_action(action, root=ROOT):
    policy=load_policy(root)
    if policy['experiment_submission']['mode']=='local_only':
        raise PolicyError('v0.4 local-only scope forbids Kaggle API actions, uploads and cloud execution.')
    if action in {'push', 'submit', 'start', 'restart'}:
        raise PolicyError('Legacy submit/start/restart is blocked; use the registered, validated package and guard_authorized_submission.')

def guard_authorized_submission(folder, root=ROOT):
    p=load_policy(root)['experiment_submission']
    if p.get('awaiting_current_run_approval'):
        raise PolicyError('The user requested approval after reviewing this concrete v0.4 package; submission remains blocked until that reply.')
    if p['mode'] not in {'agent_authorized_once','agent_submit_then_handoff'} or not p['agent_may_submit']:
        raise PolicyError('This submission has no direct user authorization.')
    if not p.get('allowed_package'):
        raise PolicyError('No validated online package is registered yet; do not submit or reuse the completed v0.3 run.')
    if Path(folder).resolve()!=(root/p['allowed_package']).resolve():
        raise PolicyError('Only the registered submission package is allowed.')
    if p['mode']=='agent_submit_then_handoff':
        package=(root/p['allowed_package']).resolve()
        if not package.is_relative_to(root.resolve()) or not package.is_dir():
            raise PolicyError('The validated submission package must exist inside the current project.')
        receipt=p.get('local_validation_receipt')
        if not receipt or not (root/receipt).resolve().is_relative_to(root.resolve()) or not (root/receipt).is_file():
            raise PolicyError('Register the local validation receipt before online execution.')
    if p['maximum_main_submissions']!=1:
        raise PolicyError('The main run is limited to one submission; automatic restart is forbidden.')
    return p

def guard_legacy_runtime(root=ROOT):
    policy = load_policy(root)
    if not policy['assets']['allow_runtime_model_or_data_download']:
        raise PolicyError('Legacy automatic-download entry is disabled. Use kaggle/online-v03/online_v03.ipynb with verified local mounts.')

def validate_online_policy(config, root=ROOT):
    p = load_policy(root)
    if p['experiment_submission']['mode']=='local_only':
        raise PolicyError('v0.4 is local preparation only; cloud/model execution is not authorized.')
    if p['experiment_submission']['mode']=='agent_submit_then_handoff' and not frozen_runtime(p,root):
        guard_authorized_submission(root/(p['experiment_submission'].get('allowed_package') or ''),root)
    if p['assets']['allow_runtime_model_or_data_download'] or p['assets']['missing_or_invalid_assets'] != 'fail_with_actionable_message':
        raise PolicyError('v0.3 requires offline assets and explicit validation failures.')
    if config['seed'] != p['research']['seed'] or config['seed'] != 42:
        raise PolicyError('The frozen seed is 42.')
    if p['research']['held_out_test_execution'] != 'forbidden_until_method_freeze':
        raise PolicyError('Held-out test must remain sealed.')
    if p['research']['smoke_questions'] != 20 or p['research']['calibration_questions'] != 200:
        raise PolicyError('The frozen partitions are 20 smoke / 200 calibration.')
    return p

def guard_local_v04(root=ROOT):
    p=load_policy(root)
    if p['experiment_submission']['mode'] not in {'local_only','agent_submit_then_handoff'} or p['research']['held_out_test_execution']!='forbidden_until_method_freeze' or p['research']['seed']!=42:
        raise PolicyError('v0.4 local preparation requires an applicable project policy, seed42, and sealed held-out test.')
    return p

def guard_model_execution(root=ROOT):
    p=load_policy(root)
    if frozen_runtime(p,root):return p
    if p['experiment_submission'].get('awaiting_current_run_approval'):
        raise PolicyError('Current v0.4 request prepares the package and asks for approval before GPU/model execution.')
    if p['experiment_submission']['mode']=='local_only':
        raise PolicyError('v0.4 is local preparation; real model inference is not authorized.')
    if p['experiment_submission']['mode']=='agent_submit_then_handoff':
        guard_authorized_submission(root/(p['experiment_submission'].get('allowed_package') or ''),root)
    return p
