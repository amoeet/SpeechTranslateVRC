import os
import tempfile
from flask import Flask, request, jsonify
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# --- Configuration ---
# Specify the model directory or model ID that AutoModel can download.
# If "SenseVoiceSmall" is a local directory, ensure it's accessible.
# If it's a model ID, AutoModel will attempt to download it.
MODEL_DIR = "SenseVoiceSmall"
# Specify the device: "cuda:0" for GPU (if available), or "cpu".
# Ensure you have the necessary CUDA drivers and PyTorch version if using GPU.
DEVICE = "cuda:0" # or "cpu"

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load FunASR Model ---
# This model is loaded once when the Flask app starts.
print(f"Loading FunASR model: {MODEL_DIR} on device: {DEVICE}...")
try:
    model = AutoModel(
        model=MODEL_DIR,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=DEVICE,
    )
    print("FunASR model loaded successfully.")
except Exception as e:
    print(f"Error loading FunASR model: {e}")
    # If the model fails to load, you might want to exit or handle this gracefully.
    # For this example, we'll let Flask start but the /transcribe endpoint will fail.
    model = None

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    API endpoint to transcribe an uploaded audio file.
    Expects a POST request with a file part named 'audio_file'.
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file part in the request"}), 400

    audio_file = request.files['audio_file']

    if audio_file.filename == '':
        return jsonify({"error": "No selected audio file"}), 400

    if audio_file:
        try:
            # Create a temporary file to save the uploaded audio
            # tempfile.NamedTemporaryFile creates a file that is deleted when closed.
            # We need to pass the filename string to funasr, so we'll manage deletion manually.
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav") # You can adjust suffix if needed
            audio_file.save(temp_audio_file.name)
            temp_audio_path = temp_audio_file.name
            temp_audio_file.close() # Close the file handle

            print(f"Temporary audio file saved at: {temp_audio_path}")

            # Perform transcription
            # The cache parameter can be an empty dict or a more sophisticated caching object
            # if you expect to process the same audio multiple times.
            res = model.generate(
                input=temp_audio_path,
                cache={},
                language="auto",  # Automatically detect language
                use_itn=True,     # Apply Inverse Text Normalization
                batch_size_s=60,  # Batch size in seconds
                merge_vad=True,   # Merge VAD segments
                merge_length_s=15 # Merge segments up to this length in seconds
            )

            if not res or "text" not in res[0]:
                return jsonify({"error": "Transcription failed or returned unexpected result"}), 500

            # Post-process the transcription for rich formatting (if applicable)
            transcribed_text = rich_transcription_postprocess(res[0]["text"])

            return jsonify({"transcription": transcribed_text})

        except Exception as e:
            print(f"Error during transcription: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up the temporary file
            if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                print(f"Temporary audio file {temp_audio_path} deleted.")
    else:
        return jsonify({"error": "Audio file processing failed"}), 400

if __name__ == '__main__':
    # Run the Flask app
    # For development, host='0.0.0.0' makes it accessible on your network.
    # For production, use a proper WSGI server like Gunicorn or uWSGI.
    app.run(host='0.0.0.0', port=5000, debug=True)
