pipeline {
	agent any
	    stages {
	        stage('Clone Repository') {
	        /* Cloning the repository to our workspace */
	        steps {
			echo 'Hello World'
	        /* checkout scm */
	        }
	   }
	   stage('Build Image') {
	        steps {
			echo 'Testing..'
	        /* docker build -t nlpproject . */
	        }
	   }
	   stage('Run Image') {
	        steps {
			echo 'Testing..'
	        /* docker run -d -p 5000:5000 nlpproject */
	        }
	   }
	   stage('Testing'){
	        steps {
	        echo 'Testing..'
	        }
	   }
     }
}
