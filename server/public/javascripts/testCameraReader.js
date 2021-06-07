const video = document.getElementById('video');
const canvas = document.getElementById('shot');
const button = document.getElementById('startCamera');

let state = 0;
let idInterval;

button.addEventListener('click', () => {
    switch (state) {
        case 0:
            accessCamera();
            state = 1;

        case 1:
            video.play();
            state = 2;
            button.innerHTML = 'Остановить чтение видеопотока';
            break;

        case 2:
            video.pause();
            state = 1;
            button.innerHTML = 'Начать чтение видеопотока';
            break;
    }
});

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia ||
    navigator.mozGetUserMedia;

function accessCamera() {
    navigator.getUserMedia({
            video: true,
            audio: false
        },

        (stream) => {
            video.srcObject = stream;
        },

        (error) => {
            console.error(error);
        });
}

video.addEventListener('play', (event) => {
    console.log(event);

    idInterval = setInterval(async() => {
        canvas.setAttribute('width', video.videoWidth / 4);
        canvas.setAttribute('height', video.videoHeight / 4);

        const data = takePicture();
        console.log('size: ' + data.length);

        sendToServer(data);
    }, 1000);
});

video.addEventListener('pause', (event) => {
    console.log(event);
    clearInterval(idInterval);
});

function takePicture() {
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const data = canvas.toDataURL('image/png');
    return data;
}

function sendToServer(data /** base64 img*/ ) {

}