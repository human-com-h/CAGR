"""Kaggle pilot: Direct/RAG1/Multihop2 traces and offline marginal gains.

Does not fit a router or execute held-out test questions.
"""
from pathlib import Path
import argparse,collections,hashlib,importlib.metadata,json,math,os,platform,re,sqlite3,string,time

ROOT=Path(__file__).resolve().parents[1]
STOPWORDS=set("a an the is are was were of in on at to for and or as by with from which what who when where how did does do has have had be been this that it its than name whose".split())

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def normalize_answer(text):
    text="".join(c for c in text.lower() if c not in string.punctuation)
    text=re.sub(r"\b(a|an|the)\b"," ",text)
    return " ".join(text.split())

def answer_metrics(prediction,references,dataset):
    pred=normalize_answer(prediction)
    result=[]
    for answer in references:
        gold=normalize_answer(answer)
        em=float(pred==gold)
        if dataset=="hotpotqa" and (pred in {"yes","no","noanswer"} or gold in {"yes","no","noanswer"}) and pred!=gold:
            f1=0.0
        else:
            p,g=pred.split(),gold.split()
            common=sum((collections.Counter(p)&collections.Counter(g)).values())
            f1=2*common/(len(p)+len(g)) if p and g else float(p==g)
        result.append((em,f1))
    return {"em":max(x[0] for x in result),"f1":max(x[1] for x in result)}

class Retriever:
    def __init__(self,path,config):
        self.connection=sqlite3.connect(f"file:{path.as_posix()}?mode=ro",uri=True)
        self.config=config

    def search(self,query):
        start=time.perf_counter()
        terms=[t for t in re.findall(r"[a-z0-9]+",query.lower()) if t not in STOPWORDS]
        terms=list(dict.fromkeys(terms))[:self.config["max_query_terms"]]
        expression=" OR ".join('"'+t+'"' for t in terms)
        rows=[]
        if expression:
            rows=self.connection.execute("SELECT doc_id,title,text,-bm25(passages,0,?,?) AS score FROM passages WHERE passages MATCH ? ORDER BY bm25(passages,0,?,?),rowid LIMIT ?",(self.config["title_weight"],self.config["text_weight"],expression,self.config["title_weight"],self.config["text_weight"],self.config["score_candidates"])).fetchall()
        elapsed=1000*(time.perf_counter()-start)
        docs=[dict(zip(["doc_id","title","text","score"],r)) for r in rows]
        feature_start=time.perf_counter()
        scores=[d["score"] for d in docs]
        mean=sum(scores)/len(scores) if scores else 0.0
        variance=sum((s-mean)**2 for s in scores)/len(scores) if scores else 0.0
        feature_ms=1000*(time.perf_counter()-feature_start)
        return docs,{"kind":"retrieval","query":query,"candidate_ids":[d["doc_id"] for d in docs],"scores":scores,"score_variance":variance,"candidate_count":len(docs),"retrieval_feature_ms":feature_ms,"query_sql_ms":elapsed,"ms":1000*(time.perf_counter()-start),"status":"ok"}

class Generator:
    def __init__(self,config,revision):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.torch=torch
        self.config=config
        model_source=config.get("local_model_path",config["repo_id"])
        offline=bool(config.get("local_model_path"))
        self.tokenizer=AutoTokenizer.from_pretrained(model_source,revision=revision,local_files_only=offline)
        self.model=AutoModelForCausalLM.from_pretrained(model_source,revision=revision,local_files_only=offline,torch_dtype=torch.float16,attn_implementation=config["attention_implementation"]).to(config["device"])
        self.model.eval()
        eos=self.model.generation_config.eos_token_id
        self.eos=set(eos if isinstance(eos,list) else [eos])
        self.calls=[]

    def generate(self,system,user,max_tokens,kind,collect_signals=True):
        torch=self.torch
        start=time.perf_counter()
        messages=[{"role":"system","content":system},{"role":"user","content":user}]
        text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        ids=self.tokenizer(text,return_tensors="pt",add_special_tokens=False)["input_ids"].to(self.config["device"])
        if ids.shape[1]>self.config["max_input_tokens"]:
            raise ValueError("Prompt exceeds frozen input budget")
        torch.cuda.reset_peak_memory_stats()
        call={"kind":kind,"input_tokens":ids.shape[1],"output_tokens":0,"signal_ms":0.0,"status":"running","prompt_sha256":hashlib.sha256(text.encode()).hexdigest()}
        self.calls.append(call)
        generated=[]
        entropies=[]
        margins=[]
        gaps=[]
        prefix_ms=0.0
        past=None
        next_ids=ids
        try:
            with torch.inference_mode():
                for step in range(max_tokens):
                    result=self.model(input_ids=next_ids,past_key_values=past,use_cache=True)
                    logits=result.logits[:,-1,:]
                    if collect_signals and step<self.config["pilot_k"]:
                        torch.cuda.synchronize()
                        stat_start=time.perf_counter()
                        logp=logits.float().log_softmax(-1)
                        probs=logp.exp()
                        h=-(probs*logp).sum(-1)
                        ptop=probs.topk(2,-1).values
                        ztop=logits.float().topk(2,-1).values
                        entropies.append(float(h.item()))
                        margins.append(float((ptop[0,0]-ptop[0,1]).item()))
                        gaps.append(float((ztop[0,0]-ztop[0,1]).item()))
                        torch.cuda.synchronize()
                        call["signal_ms"]+=1000*(time.perf_counter()-stat_start)
                    next_ids=logits.argmax(-1).reshape(1,1)
                    token=int(next_ids.item())
                    generated.append(token)
                    call["output_tokens"]=len(generated)
                    past=result.past_key_values
                    if step+1==self.config["pilot_k"] or (token in self.eos and step<self.config["pilot_k"]):
                        torch.cuda.synchronize()
                        prefix_ms=1000*(time.perf_counter()-start)
                    if token in self.eos:
                        break
            torch.cuda.synchronize()
            cache_length=past.get_seq_length() if hasattr(past,"get_seq_length") else None
            call.update({"status":"ok","ms":1000*(time.perf_counter()-start),"output_token_ids":generated,"pilot_prefix_ms":prefix_ms,"pilot_output_tokens":min(self.config["pilot_k"],len(generated)),"cache_final_sequence_length":int(cache_length) if cache_length is not None else None,"peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_gpu_reserved_bytes":torch.cuda.max_memory_reserved()})
        except Exception as exc:
            call.update({"status":"failed","ms":1000*(time.perf_counter()-start),"error_type":type(exc).__name__})
            raise
        answer=self.tokenizer.decode(generated,skip_special_tokens=True).strip()
        feature={"prefix_entropy":sum(entropies)/len(entropies) if entropies else None,"probability_margin":sum(margins)/len(margins) if margins else None,"logit_gap":sum(gaps)/len(gaps) if gaps else None,"effective_k":len(entropies),"answer_length_tokens":sum(t not in self.eos for t in generated),"token_entropies":entropies,"token_probability_margins":margins,"token_logit_gaps":gaps}
        return answer,feature,call

def evidence_prompt(question,docs,generator,config):
    evidence=[]
    for d in docs:
        tokens=generator.tokenizer.encode(d["text"],add_special_tokens=False)[:config["max_document_tokens"]]
        evidence.append("["+d["title"]+"]\n"+generator.tokenizer.decode(tokens))
    return "Evidence:\n"+"\n\n".join(evidence)+"\n\nQuestion: "+question

def cost(generation_calls,retrieval_calls):
    return {"total_input_tokens":sum(c["input_tokens"] for c in generation_calls),"total_output_tokens":sum(c["output_tokens"] for c in generation_calls),"retrieval_calls":len(retrieval_calls),"signal_ms":sum(c.get("signal_ms",0) for c in generation_calls)}

def run_question(sample,generator,retriever,config):
    q=sample["question"]
    model_cfg=config["model"]
    generation=config["generation"]
    retcfg=config["retrieval"]
    row={"sample_id":sample["sample_id"],"dataset":sample["dataset"],"fold":sample.get("fold"),"status":"running","baselines":{}}
    generator.calls=[]
    searches=[]
    def direct():
        start=time.perf_counter()
        answer,features,call=generator.generate(generation["answer_system"],"Question: "+q,model_cfg["max_answer_tokens"],"direct")
        row["baselines"]["direct"]={"answer":answer,"features":features,"generation_calls":[call],"retrieval_trace":[],"cost":{**cost([call],[]),"end_to_end_ms":1000*(time.perf_counter()-start)}}
    def rag():
        start=time.perf_counter()
        candidates,search=retriever.search(q)
        searches.append(search)
        docs=candidates[:retcfg["context_documents_per_round"]]
        answer,features,call=generator.generate(generation["answer_system"],evidence_prompt(q,docs,generator,retcfg),model_cfg["max_answer_tokens"],"rag1")
        features.update({"retrieval_score_variance":search["score_variance"],"candidate_count":search["candidate_count"]})
        row["baselines"]["rag1"]={"answer":answer,"features":features,"generation_calls":[call],"retrieval_trace":[search],"context_ids":[d["doc_id"] for d in docs],"cost":{**cost([call],[search]),"end_to_end_ms":1000*(time.perf_counter()-start)}}
        return docs
    try:
        # Balance which independent strategy runs first; escalation must follow RAG1.
        if int(sample["selection_sha256"],16)%2:
            direct()
            docs=rag()
        else:
            docs=rag()
            direct()
        first=row["baselines"]["rag1"]
        start=time.perf_counter()
        rewrite_user="Question: "+q+"\nPreliminary answer: "+first["answer"]+"\nEvidence titles: "+"; ".join(d["title"] for d in docs)
        query,_,rewrite=generator.generate(generation["rewrite_system"],rewrite_user,model_cfg["max_rewrite_tokens"],"query_rewrite",collect_signals=False)
        # An empty rewrite uses the original query, but the actual rewrite cost remains counted.
        more,search=retriever.search(query if query else q)
        search["empty_rewrite_fallback"]=not bool(query)
        searches.append(search)
        merged=list(docs)
        existing={d["doc_id"] for d in docs}
        for d in more[:retcfg["context_documents_per_round"]]:
            if d["doc_id"] not in existing:
                merged.append(d)
                existing.add(d["doc_id"])
        answer,features,final=generator.generate(generation["answer_system"],evidence_prompt(q,merged,generator,retcfg),model_cfg["max_answer_tokens"],"multihop_final")
        calls=first["generation_calls"]+[rewrite,final]
        ret=first["retrieval_trace"]+[search]
        increment_ms=1000*(time.perf_counter()-start)
        row["baselines"]["multihop2"]={"answer":answer,"features":features,"generation_calls":calls,"retrieval_trace":ret,"context_ids":[d["doc_id"] for d in merged],"cost":{**cost(calls,ret),"end_to_end_ms":first["cost"]["end_to_end_ms"]+increment_ms,"escalation_increment_ms":increment_ms,"timing_scope":"measured RAG1 segment plus measured continuation, reused baseline call"}}
        assert all(math.isfinite(b["features"]["prefix_entropy"]) for b in row["baselines"].values())
        row["status"]="ok"
    except Exception as exc:
        row.update({"status":"failed","error_type":type(exc).__name__,"partial_generation_calls":generator.calls,"partial_retrieval_trace":searches,"executed_cost":cost(generator.calls,searches)})
    return row

def summarize(rows,seed=42):
    import numpy as np
    result={"questions":len(rows),"failed":sum(r["status"]!="ok" for r in rows),"scope":"pilot calibration labels; no trained router or held-out test execution","by_dataset":{}}
    for dataset in sorted({r["dataset"] for r in rows}):
        selected=[r for r in rows if r["dataset"]==dataset]
        valid=[r for r in selected if r["status"]=="ok"]
        group={"questions":len(selected),"complete_pairs":len(valid),"failed":len(selected)-len(valid),"deltas":{},"baselines":{}}
        for name in ["direct","rag1","multihop2"]:
            if valid:
                group["baselines"][name]={key:float(np.mean([r["baselines"][name]["quality"][key] for r in valid])) for key in ["em","f1"]}
                group["baselines"][name].update({key:float(np.mean([r["baselines"][name]["cost"][key] for r in valid])) for key in ["total_input_tokens","total_output_tokens","retrieval_calls","end_to_end_ms","signal_ms"]})
        for key in ["delta01_em","delta01_f1","delta12_em","delta12_f1"]:
            values=np.array([r[key] for r in valid])
            if values.size:
                rng=np.random.default_rng(seed)
                boot=np.mean(values[rng.integers(0,len(values),size=(2000,len(values)))],axis=1)
                group["deltas"][key]={"n":len(values),"mean":float(values.mean()),"benefit_fraction":float(np.mean(values>1e-12)),"no_change_fraction":float(np.mean(np.abs(values)<=1e-12)),"harm_fraction":float(np.mean(values < -1e-12)),"bootstrap_mean_ci95":np.quantile(boot,[0.025,0.975]).tolist()}
        result["by_dataset"][dataset]=group
    return result

def main():
    from execution_policy import guard_legacy_runtime
    guard_legacy_runtime()
    parser=argparse.ArgumentParser()
    parser.add_argument("--partition",choices=["smoke","calibration"],required=True)
    args=parser.parse_args()
    import torch,yaml
    config=yaml.safe_load((ROOT/"configs/experiment.yaml").read_text(encoding="utf-8"))
    sources=json.loads((ROOT/"configs/source_lock.json").read_text())
    output=ROOT/"results"/args.partition
    output.mkdir(parents=True,exist_ok=True)
    env={"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"gpus":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],"dependencies":{p:importlib.metadata.version(p) for p in ["transformers","huggingface_hub","pyarrow","pyyaml"]},"generator_device":config["model"]["device"],"concurrency":1,"batch":1,"second_gpu_used":False}
    (output/"environment.json").write_text(json.dumps(env,indent=2)+"\n")
    if len(env["gpus"])!=2 or any("T4" not in name for name in env["gpus"]):
        raise RuntimeError("Expected two T4 GPUs; actual hardware recorded")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    generator=Generator(config["model"],sources["sources"]["generator"]["revision"])
    # Two identical warmup calls excluded from online latency but recorded separately.
    warmups=[]
    for _ in range(2):
        _,_,record=generator.generate(config["generation"]["answer_system"],"Question: What is 1 plus 1?",8,"warmup")
        warmups.append(record)
    (output/"warmup.json").write_text(json.dumps(warmups,indent=2)+"\n")
    retriever=Retriever(ROOT/"data/corpus/pilot.sqlite",config["retrieval"])
    questions=read_jsonl(ROOT/"data/splits"/(args.partition+".jsonl"))
    path=output/"predictions.jsonl"
    previous=read_jsonl(path) if path.exists() else []
    done={r["sample_id"] for r in previous}
    for i,sample in enumerate(questions):
        if sample["sample_id"] in done:
            continue
        row=run_question(sample,generator,retriever,config)
        with path.open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
        print(json.dumps({"partition":args.partition,"progress":i+1,"total":len(questions),"sample_id":sample["sample_id"],"status":row["status"]}),flush=True)
    # References are opened only after predictions are persisted.
    references={r["sample_id"]:r["answers"] for r in read_jsonl(ROOT/"data/references"/(args.partition+".jsonl"))}
    rows=read_jsonl(path)
    for row in rows:
        if row["status"]!="ok":
            continue
        for baseline in row["baselines"].values():
            baseline["quality"]=answer_metrics(baseline["answer"],references[row["sample_id"]],row["dataset"])
        for metric in ["em","f1"]:
            row["delta01_"+metric]=row["baselines"]["rag1"]["quality"][metric]-row["baselines"]["direct"]["quality"][metric]
            row["delta12_"+metric]=row["baselines"]["multihop2"]["quality"][metric]-row["baselines"]["rag1"]["quality"][metric]
    (output/"scored.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    summary=summarize(rows,seed=config["seed"])
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    if summary["failed"] or len(rows)!=len(questions):
        raise RuntimeError("Incomplete pilot; saved failures must be inspected before proceeding")

if __name__=="__main__":
    main()
