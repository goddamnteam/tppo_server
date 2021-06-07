const { isGuest } = require('../jwtAuthentication');

exports.analisation = function (req, res) {
    res.render('analisation.hbs', {
        title : 'Analisation',	/**< Main title of the page header*/
        isGuest : isGuest(req, res),
    });
}
