require('dotenv').config();

const db = require('./db');
const express = require('express');
const app = express();
const bodyParser = require('body-parser');
const cookie = require('cookie-parser');
const expressHbs = require('express-handlebars');
const hbs = require('hbs');
const { verify, isQuest } = require('./jwtAuthentication');
const { NodeSSH } = require('node-ssh');
const multer = require('multer');

const homeRouter = require("./routes/homeRouter.js");
const authRouter = require("./routes/authRouter.js");
const analisationRouter = require("./routes/analisationRouter");


app.engine("hbs", expressHbs({
    layoutsDir: "views/layouts",
    defaultLayout: "main-layout",
    extname: "hbs"
}));

app.set('view engine', 'hbs');
hbs.registerPartials(__dirname + "/views/partials");

app.use(express.static(__dirname + '/public'));
app.use(bodyParser.urlencoded({ extended: false }));
app.use(cookie());

const storageConfig = multer.diskStorage({
    destination: (req, file, cb) =>{
        cb(null, "uploads");
    },

    filename: (req, file, cb) =>{
        cb(null, file.originalname);
    }
});
const upload = multer({ storage: storageConfig });

app.use("/upload", upload.single('filedata'), function(req, res) {

    if (!req.file) res.redirect('/analisation');

    const filename = req.file.filename;
    console.log(`File ${filename} uploaded to server`);

    // file to raspberry
    sshToRecognizer(filename, function(outputFilename) {

        // output video to client
        res.download(__dirname + '/uploads/' + outputFilename);
    });
});

app.use("/analisation", verify, analisationRouter);
app.use("/auth", authRouter);
app.use("/", homeRouter);

app.use(function(req, res, next) {
    res.status(404).send("Not Found :(")
});

const server = app.listen(process.env.PORT, function() {
    console.log("Server is listening on port " + process.env.PORT);
});


function sshToRecognizer(filename, done) {
    console.log('starting ssh...');
    const ssh = new NodeSSH();

    ssh.connect({
        host: 'localhost',
        port: '10022',
        username: 'naccel',
        password: 'b4gUv1x'
    })
        .then(function() {

            const src = __dirname + '/uploads/';
            const dst = 'tppo/posenet_example/';

            // put file to raspberry
            ssh.putFile(src + filename, dst + filename)
                .then(function() {
                    console.log(`File ${filename} was uploaded`);

                    // start recognition
                    ssh.execCommand('cd ' + dst + ' && ' + 'python3 main.py ' + filename)
                        .then(function(result) {

                            console.log('STDOUT: ' + result.stdout);
                            console.log('STDERR: ' + result.stderr);

                            // get recognized file back
                            ssh.getFile(src + 'output_' + filename, dst + 'output_video.mp4')
                                .then(function(Contents) {
                                    console.log('File content is here');

                                    // remove file
                                    ssh.execCommand('rm ' + dst + filename)
                                        .then(function(result) {
                                            console.log('Removed file ' + filename);
                                        });

                                    // callback for recognized file
                                    done('output_' + filename);

                                }, function(error) {
                                    console.log('Error while getting file');
                                    console.log(error);
                                });
                        });

                }, function(error) {
                    console.log('Error while uploading');
                    console.log(error);
                });
        });
}

