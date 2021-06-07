const jwt = require('jsonwebtoken');
const db = require('../db');


exports.login = function (req, res) {
    res.render('./auth/login.hbs', {
        title : 'Login page',
        isGuest : true,
    });
};

exports.signUp = function (req, res) {
    res.render('./auth/signup.hbs', {
        title : 'Sign up page',
        isGuest : true,
    });
};

exports.logout = function (req, res) {
    res.clearCookie('jwt');
    res.redirect('/');
};

exports.postLogin = function(req, res) {

    // validate user data
    // check if user exists
    const log = req.body.login;
    const pass = req.body.password;

    console.log(`${log} ${pass}`);

    if (db.User.validate(log, pass) === false) {
        console.log("[postlogin]: incorrect log or pass");
        res.redirect('/auth/login');
        return;
    }

    db.User.findOne({ where: { login: log } })
        .then(user => {
            if(!user) {
                res.redirect('/auth/login');
                return;
            }

            if (pass !== user.password) {
                res.redirect('/auth/login');
                return;
            }

            console.log('User was logged in');
            console.log(user);
            console.log();

            attachJwtToken(log, res);

            res.redirect('/');
        })
        .catch(err => {
            console.log(err)
            res.redirect('/auth/login');
        });
};

exports.postSignUp = function(req, res) {

    const log = req.body.login;
    const pass = req.body.password;

    console.log(`${log} ${pass}`);

    if (db.User.validate(log, pass) === false) {
        console.log("[postsignup]: incorrect log or pass");
        res.redirect('/auth/signup');
        return;
    }

    db.User.create({ login: log, password: pass })
        .then(addedUser => {
            const user = {
                id: addedUser.id,
                login: addedUser.login,
                password: addedUser.password
            };

            console.log('User was signed up');
            console.log(user);
            console.log();

            attachJwtToken(log, res);

            res.redirect("/");
        })
        .catch(err => {
            console.log('Cannot sign up a user');
            // console.log(err)
            // better use res.send + view + data with errors
            res.redirect('/auth/signup');
        });
};

function attachJwtToken(login, res) {
    const token = jwt.sign(
        { login: login },
        process.env.JWT_SECRET_KEY,
        { expiresIn: process.env.JWT_EXPIRES });

    res.cookie("jwt", token, { secure: false, httpOnly: true });
}
