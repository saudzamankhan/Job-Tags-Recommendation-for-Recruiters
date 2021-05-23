/* For Jenkins pipeline script, use the commands below (bat in case of windows and ssh in case of linux)
bat 'docker build -t nlpproject .''
bat 'docker run -d -p 5000:5000 nlpproject' */

pipeline {
	    agent any

	    stages {
	        stage('Clone Repository') {
	        /* Cloning the repository to our workspace */
	        steps {
	        checkout scm
	        }
	   }
	   stage('Build Image') {
	        steps {
			bat 'docker container stop $(docker container list -aq)'
			bat 'docker container rm $(docker container ls –aq)'
	        bat 'docker build -t nlpproject .'
	        }
	   }
	   stage('Run Image') {
	        steps {
	        bat 'docker run -d -p 5000:5000 nlpproject'
	        }
	   }
	}
}