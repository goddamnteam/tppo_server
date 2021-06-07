const { isGuest } = require('../jwtAuthentication');

exports.index = function (req, res) {
    res.render('home.hbs', {
        title : 'Home page',
        isGuest : isGuest(req, res),
    });
};
