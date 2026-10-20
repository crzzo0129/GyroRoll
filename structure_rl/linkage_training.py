"""Versioned free-torso linkage curriculum and promotion rules."""
import math
import torch

REVISION = 'linkage_balance_gated_v2_20261020'
PHASE_ORDER = ('warmup', 'moving', 'tracking')
RESET_TILT = {'warmup': math.radians(.1), 'moving': math.radians(.3),
              'tracking': math.radians(.85)}
DISTRIBUTION = {'class_name': 'miniwheelbot_rl.distributions:BoundedMeanGaussianDistribution',
                'init_std': .08, 'min_std': .04, 'max_std': .15}


def posture_reward(gravity):
    roll=torch.atan2(-gravity[:,1],-gravity[:,2])
    pitch=torch.atan2(gravity[:,0],torch.linalg.vector_norm(gravity[:,1:],dim=1))
    return torch.exp(-(roll/.10).square()-(pitch/.10).square())


def limit_cost(angle, limit):
    return (angle.abs()/limit-.65).clamp_min(0).square()


def gate_cases(phase):
    if phase=='warmup':
        return [('stand',0.,0.)]
    if phase=='moving':
        return [('stand',0.,0.),('forward',.3,0.),('reverse',-.3,0.)]
    if phase=='tracking':
        return [('stand',0.,0.),('forward',1.,0.),('reverse',-1.,0.),
                ('left',.6,.3),('right',.6,-.3),
                ('reverse_left',-.6,.3),('reverse_right',-.6,-.3),
                ('reversal',.6,0.)]
    raise ValueError(phase)


def promotion_passed(report, motion_fraction=1.):
    """All cases must pass; no promotion from rewards, missing cases or short trials."""
    if report.get('controller','neural_policy')!='neural_policy':
        return False
    if report.get('motion_fraction',1.)!=motion_fraction:
        return False
    phase=report['phase']
    expected={name for name,_,_ in gate_cases(phase)}
    cases=report['cases']
    if len(cases)!=len(expected) or {c['name'] for c in cases}!=expected:
        return False
    if report['duration_s'] < 10 or report['trials_per_case'] < 16:
        return False
    for case in cases:
        values=[case['survival_rate'],case['pass_rate'],case['max_closure_error_m']]
        if not all(math.isfinite(v) for v in values):
            return False
        if case['survival_rate'] < .9 or case['pass_rate'] < .8 or case['max_closure_error_m'] >= .001:
            return False
    return True


def candidate_acceptable(candidate, baseline):
    """Reject nonfinite physics and loss of previously demonstrated capabilities."""
    if candidate.get('controller','neural_policy')!='neural_policy':
        return False
    if (candidate['phase'],candidate['duration_s'],candidate['trials_per_case']) != (
            baseline['phase'],baseline['duration_s'],baseline['trials_per_case']):
        return False
    if candidate.get('motion_fraction',1.)!=baseline.get('motion_fraction',1.):
        return False
    old={c['name']:c for c in baseline['cases']}
    if {c['name'] for c in candidate['cases']}!=set(old):
        return False
    for case in candidate['cases']:
        previous=old[case['name']]
        for key in ('survival_rate','pass_rate','mean_survival_s','max_closure_error_m'):
            if not math.isfinite(case[key]):
                return False
        if case['max_closure_error_m']>=.001:
            return False
        # No case may lose successful trials, even if the aggregate improves.
        if any(case[k]+1e-8<previous[k] for k in ('survival_rate','pass_rate')):
            return False
        if case['mean_survival_s']+.05<previous['mean_survival_s']:
            return False
    return True


def mixed_candidate_acceptable(candidate, baseline):
    """Permit tradeoffs among unfinished maneuvers while retaining passed skills."""
    if candidate.get('controller','neural_policy')!='neural_policy':
        return False
    for key in ('phase','duration_s','trials_per_case','initial_tilt_bound_deg'):
        if candidate.get(key)!=baseline.get(key):
            return False
    old={c['name']:c for c in baseline['cases']}
    if len(candidate['cases'])!=len(old) or {c['name'] for c in candidate['cases']}!=set(old):
        return False
    def score(case, duration):
        return case['survival_rate']+case['pass_rate']+case['mean_survival_s']/duration
    previous_score=current_score=0.
    for case in candidate['cases']:
        previous=old[case['name']]
        keys=('survival_rate','pass_rate','mean_survival_s','max_closure_error_m')
        if not all(math.isfinite(c[k]) for c in (case,previous) for k in keys):
            return False
        if case['max_closure_error_m']>=.001:
            return False
        if previous['survival_rate']>=.9 and previous['pass_rate']>=.8:
            if case['survival_rate']<.9 or case['pass_rate']<.8:
                return False
        current_score+=score(case,candidate['duration_s'])
        previous_score+=score(previous,baseline['duration_s'])
    return current_score+1e-8>=previous_score
