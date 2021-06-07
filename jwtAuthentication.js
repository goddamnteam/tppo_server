const jwt = require('jsonwebtoken');

exports.verify = function(req, res, next) {
    let accessToken = req.cookies.jwt;

    if (!accessToken){
        console.log('Cookie problem: ');
        console.log(req.cookies);
        return res.status(403).redirect('/auth/login');
    }

    let payload;
    try {
        // use the jwt.verify method to verify the access token
        // throws an error if the token has expired or has an invalid signature
        payload = jwt.verify(accessToken, process.env.JWT_SECRET_KEY);
        console.log(payload);
        next();
    }
    catch(e) {
        // if an error occured return request unauthorized error
        return res.status(401).redirect('/auth/login');
    }
}


exports.isGuest = function(req, res) {
    let accessToken = req.cookies.jwt;

    if (!accessToken){
        return true;
    }

    let payload;
    try {
        payload = jwt.verify(accessToken, process.env.JWT_SECRET_KEY);
        return false;
    }
    catch(e) {
        return true;
    }

    return true;
}

