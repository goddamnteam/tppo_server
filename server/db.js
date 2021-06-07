const Sequelize = require("sequelize");
const mysql = require('mysql2');

module.exports = db = {};

initialize();

function initialize() {
    // create db if it doesn't already exist
    const name = process.env.DB_NAME;
    const host = process.env.DB_HOST;
    const port = process.env.DB_PORT;
    const user = process.env.DB_USER;
    const pass = process.env.DB_PASSWORD;

    const connection = mysql.createConnection({
        host: host,
        user: user,
        password: pass
    });

    connection.query(`CREATE DATABASE IF NOT EXISTS \`${name}\`;`);

    // connect to db
    const sequelize = new Sequelize(name, user, pass, {
        host: host,
        dialect: 'mysql',
        define: {
            timestamps: false
        }
    });

    // load models
    db.User = require('./models/user')(sequelize);

    // sync all models with database
    sequelize.sync().catch(err => console.log(err));
}
