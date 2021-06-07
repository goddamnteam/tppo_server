const express = require('express');
const analisationController = require('../controllers/analisationController.js');
const analisationRouter = express.Router();

analisationRouter.use('/', analisationController.analisation);

module.exports = analisationRouter;
