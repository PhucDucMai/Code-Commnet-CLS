Object subclass: #RetryPolicy
	instanceVariableNames: 'maxAttempts delay'
	classVariableNames: ''
	package: 'Demo-Network'!
!RetryPolicy commentStamp: 'demo 10/23/2026 09:00' prior: 0!
I represent a strategy for retrying failed network requests.
I know how many attempts are allowed and how long to wait between two attempts.
I work with HttpClient, which asks me whether a failed request should be sent again.
Example: (RetryPolicy maxAttempts: 3 delay: 2 seconds) shouldRetry: anError!

!RetryPolicy methodsFor: 'testing'!
shouldRetry: anError
	^ anError isNetworkError and: [ maxAttempts > 0 ]! !
