from transformers import pipeline
import glob
import time

print('Downloading/loading pretrained model: dima806/deepfake_vs_real_image_detection ...')
pipe = pipeline('image-classification', model='dima806/deepfake_vs_real_image_detection', device=-1)
print('Loaded successfully!')

real_files = glob.glob('data/image_eval/real/*.*')[:15]
fake_files = glob.glob('data/image_eval/fake/*.*')[:15]

correct = 0
total = len(real_files) + len(fake_files)

print('\n--- Testing Real Samples ---')
for f in real_files:
    res = pipe(f)
    top = res[0]
    is_real = top['label'].upper() == 'REAL'
    if is_real:
        correct += 1
    print(f'{f}: {top["label"]} ({top["score"]:.2%}) -> {"PASS" if is_real else "FAIL"}')

print('\n--- Testing Fake Samples ---')
for f in fake_files:
    res = pipe(f)
    top = res[0]
    is_fake = top['label'].upper() == 'FAKE'
    if is_fake:
        correct += 1
    print(f'{f}: {top["label"]} ({top["score"]:.2%}) -> {"PASS" if is_fake else "FAIL"}')

print(f'\nOverall Accuracy: {correct}/{total} = {correct/total:.1%}')
