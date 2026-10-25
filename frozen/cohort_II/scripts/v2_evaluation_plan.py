"""Fixed scheduling and coverage for the prospective light design; no execution."""
import random,math
METHODS=('direct','always_rag','always_multihop','matched_binary_v2','targ_style','sugar_l_style_budgeted','cagr_v1','cagr_v2')

def method_order(index):
    base=list(METHODS);random.Random(42).shuffle(base);shift=index%len(base)
    return base[shift:]+base[:shift]

def pair_orders(index,pairs):
    if pairs not in [2,4]:raise ValueError('Only preregistered two or four balanced pairs')
    return [('on','off') if (index+j)%2==0 else ('off','on') for j in range(pairs)]

def shards(questions):
    if len(questions)!=1000 or len({r['sample_id'] for r in questions})!=1000:raise ValueError('Expected exactly 1000 unique question IDs')
    if any(set(r)-{'sample_id','dataset','question','source_split','source_row','selection_index'} for r in questions):raise ValueError('Question schema must exclude references and routing labels')
    groups=[[r for r in questions if r['dataset']==ds] for ds in ['hotpotqa','triviaqa']]
    if any(len(g)!=500 for g in groups):raise ValueError('Expected 500 questions per dataset')
    return [[r for pair in zip(groups[0][i*250:(i+1)*250],groups[1][i*250:(i+1)*250]) for r in pair] for i in range(2)]

def schedule(questions,pairs=2):
    result=[]
    for i,r in enumerate(questions):
        for method in method_order(i):result.append({'sample_id':r['sample_id'],'method':method,'scope':'normal'})
        for j,order in enumerate(pair_orders(i,pairs)):
            for onoff in order:result.append({'sample_id':r['sample_id'],'method':'cagr_v2','scope':'technical','pair':j,'features':onoff})
    return result

def question_overhead(pair_times):
    if not pair_times:raise ValueError('Missing fixed pairs')
    if any(not math.isfinite(on) or not math.isfinite(off) or on<0 or off<=0 for on,off in pair_times):raise ValueError('Invalid wall time')
    return sum((on-off)/off for on,off in pair_times)/len(pair_times)
