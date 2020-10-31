let model;

//openCvReady is the function that will be executed when the opencv.js file is loaded
function openCvReady() {
  cv['onRuntimeInitialized']= ()=>{
    // The variable video extracts the video the video element
    let video = document.getElementById("cam_input"); // video is the id of video tag
    // navigator.mediaDevices.getUserMedia is used to access the webcam
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then(function(stream) {
        video.srcObject = stream;
        video.play();
    })
    .catch(function(err) {
        console.log("An error occurred! " + err);
    });

    //saving the URL of the hosted model to modelURL
    let modelURL='https://teachablemachine.withgoogle.com/models/9NRc5H8KQ/';
    //src and dst holds the source and destination image matrix
    let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
    //gray holds the grayscale image of the src
    let gray = new cv.Mat();
    //cap holds the current frame of the video
    let cap = new cv.VideoCapture(cam_input);
    //RectVector is used to hold the vectors of different faces
    let faces = new cv.RectVector();
    let predictions="Detecting..."
    //classifier holds the classifier object
    let classifier = new cv.CascadeClassifier();
    let utils = new Utils('errorMessage');
    //crop holds the ROI of face
    let crop=new cv.Mat(video.height, video.width, cv.CV_8UC1);
    let dsize = new cv.Size(224, 224);

    // Loading the haar cascade face detector
    let faceCascadeFile = 'haarcascade_frontalface_default.xml'; // path to xml
    utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
    classifier.load(faceCascadeFile); // in the callback, load the cascade from file 
});


//Loading the model with async as loading the model may take few miliseconds
//The function dont take and return anything
//the model holds the classifier model
(async () => {
   model = await ml5.imageClassifier(modelURL+'model.json',video=video)
   console.log(model)
 })();

// functions takes the input as the canvas element and assign the predicted value to predictions 
 async function predict(img){
     predictions=await model.classify(img)
     return predictions
 }

    const FPS = 24;
    // processvideo will be executed recurrsively 
    function processVideo() {
        let begin = Date.now();
        cap.read(src);
        src.copyTo(dst);
        cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0); // converting to grayscale
        try{
            classifier.detectMultiScale(gray, faces, 1.1, 3, 0);// detecting the face
            console.log(faces.size());
        }catch(err){
            console.log(err);
        }
        //iterating over all the detected faces
        for (let i = 0; i < faces.size(); ++i) {
            let face = faces.get(i);
            // filtering out the boxes with the area of less than 45000
            if(face.width*face.height <40000){continue;} 
            let point1 = new cv.Point(face.x, face.y);
            let point2 = new cv.Point(face.x + face.width, face.y + face.height);
            // creating the bounding box
            cv.rectangle(dst, point1, point2, [51, 255, 255, 255],3);
            //creating a rect element that can be used to extract
            let cutrect=new cv.Rect(face.x,face.y,face.width,face.height)
            //extracting the ROI
            crop=dst.roi(cutrect)
           //creating a canvas element and convert the image matrix to canvas element
            c_temp=document.createElement('canvas')
            c_temp.setAttribute("width",224)
            c_temp.setAttribute("height",224)
            ctx_temp=c_temp.getContext('2d')
            let imgData = new ImageData(new Uint8ClampedArray(crop.data), crop.cols, crop.rows);
            ctx_temp.putImageData(imgData,0,0)
            

            //Making predictions by passing the canvas element to the predict function
            predict(video)
           // console.log("Detected face ",i)
            //console.log(ctx_temp)
            console.log(predictions)
            //adding the text above the bounding boxes
            cv.putText(dst,String(predictions[0].label).toUpperCase(),{x:face.x,y:face.y-20},1,3,[255, 128, 0, 255],4);
           
        }
      

        // showing the final output
        cv.imshow("canvas_output", dst);
       
        let delay = 1000/FPS - (Date.now() - begin);
        setTimeout(processVideo, delay);
}
// schedule first one.
setTimeout(processVideo, 0);
  };
}