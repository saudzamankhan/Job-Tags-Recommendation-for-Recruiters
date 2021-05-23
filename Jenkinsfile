/* For Jenkins pipeline script, use the commands below (bat in case of windows and ssh in case of linux)
bat 'docker build -t nlpproject .''
bat 'docker run -d -p 5000:5000 nlpproject' */

pipeline {
	    agent any

        environment {
        CMD = "curl -I http://localhost:5000/"
		}

	    stages {
	        stage('Clone Repository') {
	        /* Cloning the repository to our workspace */
	        steps {
	        checkout scm
	        }
	   }
	   stage('Build Image') {
	        steps {
	        bat 'docker build -t nlpproject .'
	        }
	   }
	   stage('Run Image') {
	        steps {
	        bat 'docker run -d -p 5000:5000 nlpproject'
	        }
	   }
	   stage('Testing'){
	        steps {
				script{
                    bat "${CMD} > commandResult"
                    env.status = readFile('commandResult').trim()
					bat "echo ${env.status}"
                    if (env.status == '200') {
                        currentBuild.result = "SUCCESS"
                    }
                    else {
                        currentBuild.result = "FAILURE"
                    }
                }
	        }
	   }
   }
}