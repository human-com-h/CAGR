"""Evidence-conditioned upgrade ablation with frozen, shared RAG1 answers."""
from pathlib import Path
import argparse,copy,hashlib,importlib.metadata,json,math,platform,time
import numpy as np
from run_baselines import Generator,Retriever,read_jsonl,answer_metrics,evidence_prompt,cost,summarize

ROOT=Path(__file__).resolve().parents[1]

def rewrite_prompt(question,first,docs,generator,study,mode):
    prefix="Question: "+question+"\nPreliminary answer (may be wrong): "+first["answer"]
    if mode=="titles_control":
        # Preserve the old prompt exactly, including its wording.
        return "Question: "+question+"\nPreliminary answer: "+first["answer"]+"\nEvidence titles: "+"; ".join(d["title"] for d in docs)
    passages=[]
    for doc in docs[:study["rewrite_evidence_documents"]]:
        ids=generator.tokenizer.encode(doc["text"],add_special_tokens=False)[:study["rewrite_tokens_per_document"]]
        passages.append("["+doc["title"]+"]\n"+generator.tokenizer.decode(ids))
    return prefix+"\nEvidence:\n"+"\n\n".join(passages)+"\nMissing evidence search query:"

def tracked_search(retriever,query,ledger,scope):
    start=time.perf_counter()
    try:
        docs,trace=retriever.search(query)
        trace["study_scope"]=scope
        ledger.append(trace)
        return docs,trace
    except Exception as exc:
        ledger.append({"query":query,"study_scope":scope,"status":"failed","ms":1000*(time.perf_counter()-start),"error_type":type(exc).__name__})
        raise

def finish_upgrade(plan,first,generator,config,study,stop_on_no_new):
    decide=time.perf_counter()
    stop=stop_on_no_new and not plan["additions"]
    decision_ms=1000*(time.perf_counter()-decide)
    if stop:
        answer=first["answer"]
        features=copy.deepcopy(first["features"])
        calls=[plan["rewrite"]]
        increment_ms=plan["after_search_ms"]+decision_ms
    else:
        answer,features,final=generator.generate(config["generation"]["answer_system"],evidence_prompt(plan["question"],plan["merged"],generator,config["retrieval"]),config["model"]["max_answer_tokens"],"final_"+plan["mode"])
        calls=[plan["rewrite"],final]
        increment_ms=1000*(time.perf_counter()-plan["start"])
    all_calls=first["generation_calls"]+calls
    searches=first["retrieval_trace"]+[plan["search"]]
    return {"answer":answer,"features":features,"generation_calls":all_calls,"retrieval_trace":searches,"context_ids":[d["doc_id"] for d in plan["merged"]],"new_document_ids":[d["doc_id"] for d in plan["additions"]],"rewrite_evidence_ids":[d["doc_id"] for d in plan["initial_docs"]],"query_mode":plan["mode"],"stopped_without_regeneration":bool(stop),"stop_reason":"no_new_documents" if stop else None,"cost":{**cost(all_calls,searches),"end_to_end_ms":first["cost"]["end_to_end_ms"]+increment_ms,"escalation_increment_ms":increment_ms,"decision_ms":decision_ms,"timing_scope":study["timing_scope"]}}

def run_variant(question,first,initial_docs,generator,retriever,config,study,mode,ledger):
    start=time.perf_counter()
    try:
        system=config["generation"]["rewrite_system"] if mode=="titles_control" else study["gap_rewrite_system"]
        user=rewrite_prompt(question,first,initial_docs,generator,study,mode)
        query,_,rewrite=generator.generate(system,user,config["model"]["max_rewrite_tokens"],"query_rewrite_"+mode,collect_signals=False)
        candidates,search=tracked_search(retriever,query if query else question,ledger,"variant_"+mode)
        search["empty_rewrite_fallback"]=not bool(query)
        known={d["doc_id"] for d in initial_docs}
        additions=[]
        # Match v0.1 top-five selection; do not alter retriever or candidate budget.
        for doc in candidates[:config["retrieval"]["context_documents_per_round"]]:
            if doc["doc_id"] not in known:
                additions.append(doc)
                known.add(doc["doc_id"])
        after_search_ms=1000*(time.perf_counter()-start)
        plan={"question":question,"initial_docs":initial_docs,"merged":initial_docs+additions,"additions":additions,"mode":mode,"start":start,"after_search_ms":after_search_ms,"rewrite":rewrite,"search":search}
        if mode=="titles_control":
            return finish_upgrade(plan,first,generator,config,study,False),None
        stopped=finish_upgrade(plan,first,generator,config,study,True)
        always=finish_upgrade(plan,first,generator,config,study,False) if stopped["stopped_without_regeneration"] else copy.deepcopy(stopped)
        return always,stopped
    except Exception:
        raise

def run_sample(sample,initial,generator,retriever,config,study):
    row={"sample_id":sample["sample_id"],"dataset":sample["dataset"],"fold":sample.get("fold"),"status":"running","baselines":copy.deepcopy(initial["baselines"]),"variants":{}}
    generator.calls=[]
    ledger=[]
    start=time.perf_counter()
    try:
        first=row["baselines"]["rag1"]
        # Reconstruct already-fetched documents as offline preparation, not a new online call.
        docs,trace=tracked_search(retriever,sample["question"],ledger,"offline_reconstruction")
        prepared=docs[:config["retrieval"]["context_documents_per_round"]]
        assert [d["doc_id"] for d in prepared]==first["context_ids"],"Corpus reconstruction differs from frozen RAG1"
        row["reconstruction"]={"ms":trace["ms"],"scope":"offline preparation, excluded from deployment replay cost","same_context_ids":True}
        # Only whitelisted runtime fields reach the decision path; no labels, Direct or future answers.
        runtime_first={k:first[k] for k in ["answer","features","generation_calls","retrieval_trace","context_ids","cost"]}
        order=["titles_control","evidence_gap"]
        if int(sample["selection_sha256"],16)%2:
            order.reverse()
        row["variant_execution_order"]=order
        for mode in order:
            result,stopped=run_variant(sample["question"],runtime_first,prepared,generator,retriever,config,study,mode,ledger)
            if mode=="titles_control":
                row["variants"]["titles_control"]=result
            else:
                row["variants"]["evidence_always"]=result
                row["variants"]["evidence_novelty_stop"]=stopped or copy.deepcopy(result)
        row["baselines"]["multihop2"]=row["variants"][study["selected_for_development"]]
        row["study_generated_call_count"]=len(generator.calls)
        # Offline ablations may execute a final call that the stop policy would skip.
        row["study_searches"]=ledger
        row["study_executed_cost"]={**cost(generator.calls,ledger),"end_to_end_ms":1000*(time.perf_counter()-start),"scope":"offline study total including reconstruction; variant traces represent distinct deployment policies"}
        row["status"]="ok"
    except Exception as exc:
        row.update({"status":"failed","error_type":type(exc).__name__,"partial_generation_calls":generator.calls,"partial_retrieval_trace":ledger,"executed_cost":{**cost(generator.calls,ledger),"end_to_end_ms":1000*(time.perf_counter()-start)}})
    return row

def paired_summary(rows,old,seed=42):
    summary={"scope":"paired development ablations; no test or online CAGR evaluation","by_dataset":{}}
    for dataset in sorted({r["dataset"] for r in rows}):
        selected=[r for r in rows if r["dataset"]==dataset and r["status"]=="ok"]
        group={"n":len(selected),"variants":{}}
        for name in ["titles_control","evidence_always","evidence_novelty_stop"]:
            current=[r["variants"][name] for r in selected]
            diffs=np.array([r["variants"][name]["quality"]["f1"]-r["baselines"]["rag1"]["quality"]["f1"] for r in selected])
            against_current_titles=np.array([r["variants"][name]["quality"]["f1"]-r["variants"]["titles_control"]["quality"]["f1"] for r in selected])
            against_old=np.array([r["variants"][name]["quality"]["f1"]-old[r["sample_id"]]["multihop2"]["quality"]["f1"] for r in selected])
            rng=np.random.default_rng(seed)
            boot=np.mean(diffs[rng.integers(0,len(diffs),size=(2000,len(diffs)))],axis=1) if len(diffs) else np.array([0.0])
            group["variants"][name]={"em":float(np.mean([v["quality"]["em"] for v in current])),"f1":float(np.mean([v["quality"]["f1"] for v in current])),"delta12_f1_mean":float(diffs.mean()),"delta12_f1_ci95":np.quantile(boot,[.025,.975]).tolist(),"beneficial":int(sum(diffs>1e-12)),"harmful":int(sum(diffs< -1e-12)),"delta_f1_vs_same_run_titles":float(against_current_titles.mean()),"delta_f1_vs_v01":float(against_old.mean()),"mean_increment_ms":float(np.mean([v["cost"]["escalation_increment_ms"] for v in current])),"mean_input_tokens":float(np.mean([v["cost"]["total_input_tokens"] for v in current])),"mean_output_tokens":float(np.mean([v["cost"]["total_output_tokens"] for v in current])),"stopped_without_regeneration":sum(v["stopped_without_regeneration"] for v in current),"same_answer_as_v01":sum(v["answer"]==old[r["sample_id"]]["multihop2"]["answer"] for r,v in zip(selected,current))}
        summary["by_dataset"][dataset]=group
    return summary

def main():
    from execution_policy import guard_legacy_runtime
    guard_legacy_runtime()
    parser=argparse.ArgumentParser()
    parser.add_argument("--partition",choices=["smoke","calibration"],required=True)
    args=parser.parse_args()
    import torch,yaml
    config=yaml.safe_load((ROOT/"configs/experiment.yaml").read_text())
    study=yaml.safe_load((ROOT/"configs/upgrade_v02.yaml").read_text())
    runtime=ROOT/"configs/runtime_paths.json"
    paths=json.loads(runtime.read_text()) if runtime.exists() else {}
    if paths.get("model_path"):
        config["model"]["local_model_path"]=paths["model_path"]
    sources=json.loads((ROOT/"configs/source_lock.json").read_text())
    output=ROOT/"results/upgrade-v02"/args.partition
    output.mkdir(parents=True,exist_ok=True)
    environment={"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"gpus":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],"dependencies":{p:importlib.metadata.version(p) for p in ["transformers","huggingface_hub","pyarrow","pyyaml"]},"generator_device":"cuda:0","initial_generation_rerun":False,"runtime_inputs":paths,"concurrency":1,"batch":1}
    (output/"environment.json").write_text(json.dumps(environment,indent=2)+"\n")
    if len(environment["gpus"])!=2 or any("T4" not in n for n in environment["gpus"]):
        raise RuntimeError("Expected two T4 GPUs")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    generator=Generator(config["model"],sources["sources"]["generator"]["revision"])
    warmups=[]
    for _ in range(2):
        _,features,call=generator.generate(config["generation"]["answer_system"],"Question: What is 1 plus 1?",8,"warmup")
        warmups.append({"features":features,"call":call})
    (output/"warmup.json").write_text(json.dumps(warmups,indent=2)+"\n")
    retriever=Retriever(Path(paths.get("corpus_path",str(ROOT/"data/corpus/pilot.sqlite"))),config["retrieval"])
    samples=read_jsonl(ROOT/"data/splits"/(args.partition+".jsonl"))
    initial={r["sample_id"]:r for r in read_jsonl(ROOT/"data/prior"/(args.partition+"_initial.jsonl"))}
    path=output/"predictions.jsonl"
    done={r["sample_id"] for r in read_jsonl(path)} if path.exists() else set()
    for i,sample in enumerate(samples):
        if sample["sample_id"] in done:
            continue
        row=run_sample(sample,initial[sample["sample_id"]],generator,retriever,config,study)
        with path.open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
        print(json.dumps({"partition":args.partition,"progress":i+1,"total":len(samples),"status":row["status"]}),flush=True)
    # References and historical future-branch results are read only in offline scoring.
    references={r["sample_id"]:r["answers"] for r in read_jsonl(ROOT/"data/references"/(args.partition+".jsonl"))}
    old={r["sample_id"]:r["baselines"] for r in read_jsonl(ROOT/"data/prior"/(args.partition+"_controls.jsonl"))}
    rows=read_jsonl(path)
    for row in rows:
        if row["status"]!="ok":
            continue
        for baseline in [*row["baselines"].values(),*row["variants"].values()]:
            baseline["quality"]=answer_metrics(baseline["answer"],references[row["sample_id"]],row["dataset"])
        for metric in ["em","f1"]:
            row["delta01_"+metric]=row["baselines"]["rag1"]["quality"][metric]-row["baselines"]["direct"]["quality"][metric]
            row["delta12_"+metric]=row["baselines"]["multihop2"]["quality"][metric]-row["baselines"]["rag1"]["quality"][metric]
    (output/"scored.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    summary=summarize(rows,seed=study["bootstrap_seed"])
    summary.update({"scope":"upgrade-v02 paired study with reused v01 initial answers","test_executed":False})
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    if summary["failed"] or len(rows)!=len(samples):
        raise RuntimeError("Failed samples saved; study needs inspection")
    comparison=paired_summary(rows,old,seed=study["bootstrap_seed"])
    (output/"comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
    print(json.dumps(comparison),flush=True)

if __name__=="__main__":
    main()
