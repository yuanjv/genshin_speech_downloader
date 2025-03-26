URL = "https://bbs.mihoyo.com/ys/obc/content/4073/detail"
OUTPUT_DIR = "/tmp/audio"

person_id=URL.split('/')[-2]

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import librosa

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device,torch_dtype="cpu",torch.float32

model_id = "openai/whisper-large-v3"  # Note: Corrected model ID from original

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)


dir=os.path.join(OUTPUT_DIR,person_id)
mp3_dir=os.path.join(dir,'E')


results = []
for filename in os.listdir(mp3_dir):
    if filename.lower().endswith('.mp3'):
        file_path = os.path.join(mp3_dir, filename)
        
        # Load audio with librosa (properly resampled)
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        
        # Process with timestamp generation
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            return_timestamps="word",  # Changed from True to "word" for word-level timestamps
            generate_kwargs={
                "task": "transcribe",
                "language": "english",  # Specify your audio's language
            },
        )
        result = pipe(
            {"array": audio, "sampling_rate": sr},
            return_timestamps="word",  # Ensure timestamp generation
        )
        
        del pipe
        
        results.append(
           (int(os.path.splitext(filename)[0]),result["text"])
        )

results=sorted(results)

with open(os.path.join(mp3_dir,"index.csv"),"w") as f:
    for idx,text in results:
        f.write(f"{idx},{text}\n")
