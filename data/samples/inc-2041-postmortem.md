# INC-2041 Post-mortem: card payments outage

Author: Site Reliability Engineering team, Corvane Pay. Date: 18 March 2026. Status: final.

## Summary

On 14 March 2026 card payments failed for 47 minutes, from 14:05 to 14:52 UTC. About 12% of all
card transactions in that window were declined with error code GW-503. Bank transfers and wallet
payments were not affected.

## Timeline (UTC)

- 14:05 - The TLS certificate of the internal payment gateway expired. The gateway started
  rejecting connections from the card processing service.
- 14:09 - The alert "card-auth error rate above 5%" fired and paged the on-call engineer.
- 14:31 - The expired certificate was identified as the cause.
- 14:48 - A renewed certificate was deployed to all gateway nodes.
- 14:52 - Error rates returned to normal and the incident was declared resolved.

## Root cause

The certificate had been issued manually in March 2025. The automated renewal job, introduced in
June 2025, only manages certificates created after that date, so this one was never renewed.
No alert existed for certificates close to expiry.

## Action items

1. Move all gateway certificates to automated renewal. Owner: Platform team, due 31 March 2026.
2. Alert 14 days before any certificate expires. Owner: SRE team, due 4 April 2026.
3. Require a status page update within 10 minutes of detection. Owner: Incident management.
