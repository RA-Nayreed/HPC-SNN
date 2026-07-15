from .shd_lif import AudioLIFSNN
def make_audio_snn(dataset,attention="none",lambda_=1e-2,dropout=.4):
    if dataset=="shd": return AudioLIFSNN(140,(256,256),20,attention,lambda_,dropout)
    if dataset=="ssc": return AudioLIFSNN(140,(128,128),35,attention,lambda_,dropout)
    if dataset=="dcls": raise RuntimeError("DCLS requires the optional verified dcls dependency; no fallback is provided")
    raise ValueError(f"unsupported dataset: {dataset}")
