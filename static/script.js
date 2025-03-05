let mediaRecorder;
let audioChunks = [];

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();

            mediaRecorder.addEventListener("dataavailable", event => {
                audioChunks.push(event.data);
            });
        })
        .catch(error => {
            alert("Microphone access denied! Please allow access.");
        });
}

function stopRecording() {
    if (mediaRecorder) {
        mediaRecorder.stop();
        mediaRecorder.addEventListener("stop", () => {
            const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            const audioFile = new File([audioBlob], "recorded_audio.wav", { type: "audio/wav" });

            const formData = new FormData();
            formData.append("audio", audioFile);

            fetch("/predict", {
                method: "POST",
                body: formData,
            });
        });
    }
}


function playAudio(audioUrl) {
    let audio = new Audio(audioUrl);
    audio.play();
}
