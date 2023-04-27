from pydub import AudioSegment
import os

def split_wavfile(filename, clip_length=5):
    sound = AudioSegment.from_file(filename)

    num_clips = int(sound.duration_seconds // clip_length)
    for i in range(num_clips):
        clip_id = str(i+1).zfill(3) # pad clip_id with zeros to 3 digits
        clip_start = i * clip_length * 1000
        clip_end = clip_start + clip_length * 1000

        # create a new audio segment for the clip
        clip = sound[clip_start:clip_end]

        # save the clip to a new file
        new_filename = os.path.splitext(filename)[0] + '_' + clip_id + '.wav'
        clip.export(new_filename, format='wav')

    print(f"{num_clips} clips saved.")
split_wavfile("train.wav")