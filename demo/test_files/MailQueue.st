Object subclass: #MailQueue
	instanceVariableNames: 'pending sender'
	classVariableNames: ''
	package: 'Demo-Mail'!
!MailQueue commentStamp: 'demo 10/23/2026 10:00' prior: 0!
I represent a queue of e-mails waiting to be sent.
I keep the pending messages and hand them to a MailSender in batches.
Never call flush twice in a row, a message could be sent twice!!
Example: MailQueue new add: aMail; flush!

Object subclass: #MailSender
	instanceVariableNames: 'smtpHost'
	classVariableNames: ''
	package: 'Demo-Mail'!
!MailSender commentStamp: 'demo 10/23/2026 10:00' prior: 0!
I send e-mails through an SMTP server.
I am used by MailQueue.!

!MailQueue methodsFor: 'sending'!
flush
	sender sendAll: pending.
	pending := OrderedCollection new! !
