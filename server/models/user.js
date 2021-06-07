const Sequelize = require("sequelize");

module.exports = function(sequelize) {
    const User = sequelize.define("user", {
        id: {
            type: Sequelize.INTEGER,
            autoIncrement: true,
            primaryKey: true,
            allowNull: false
        },

        login: {
            type: Sequelize.STRING,
            allowNull: false,
            unique: true
        },

        password: {
            type: Sequelize.STRING,
            allowNull: false
        }
    });

    User.validate = function(login, password) {
        return validateLogin(login) && validatePassword(password);
    }

    return User;
}

function validateLogin(login) {
    if (!login) return false;

    login = login.trim();
    return login !== '' && login.length > 0 && login.length < 10;
}

function validatePassword(password) {
    if (!password) return false;

    password = password.trim();
    return password !== '' && password.length > 0 && password.length < 255;
}