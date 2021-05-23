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
	        'docker build -t nlpproject .'
	        }
	   }
	   stage('Run Image') {
	        steps {
	        sh 'docker run -d -p 5000:5000 nlpproject'
	        }
	   }
	   stage('Testing'){
	        steps {
	        echo 'Testing..'
	        }
	   }
     }
}
