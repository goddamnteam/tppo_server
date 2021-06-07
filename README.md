# tppo_server

Server does following steps:
1) take a video from a client
2) give the video to the raspberry via SSH
3) wait from raspberry the recognized video and send it back to the client

Build and launch:
1) clone the repo on kappa.cs.petrsu.ru server
2) make sure you have mysql server started
3) set variables in .env file
4) npm install - download all dependencies
5) npm start - start server
