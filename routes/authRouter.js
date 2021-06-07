const express = require('express');
const authController = require('../controllers/authController.js');
const authRouter = express.Router();
const { verify, isGuest } = require('../jwtAuthentication');

authRouter.use('/post-login', checkGuest, authController.postLogin);
authRouter.use('/post-signup', checkGuest, authController.postSignUp);
authRouter.use('/logout', authController.logout);
authRouter.use('/login', checkGuest, authController.login);
authRouter.use('/signup', checkGuest, authController.signUp);

function checkGuest(req, res, next) {
    if (isGuest(req, res)) {
        next();
    }
    else {
        res.redirect('/');
    }
}

module.exports = authRouter;
