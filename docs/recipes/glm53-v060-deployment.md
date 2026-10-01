# GB10 deployment: v0.6.0 + temporary decisions + Q4MSE

Base v0.6.0 c4646171139ee8a3c38103eaa1699dad226ec12b.
PR127 head39de4c9 merged preserving authorship (33c5709), conflicts resolved
with v0.6 multi-fill scheduler plus queued idle scoring, GLM idle bell before
score header. This is our temporary port, NOT maintainer's announced0.6.1 port.
Q4MSE port cherry-picked (be01139), memory rewrite retained with optional MTP
accounting. PR140 and128/129/131/132/133/134 already upstream: not reapplied.

Both hosts mount /var/tmp/tf-v060/src at /opt/tf, existing local image
ab68ecbfcb2bc1ace4845bc1c8e768a36bdb15b94f8f0db920b61563af4d6685.
Source override means this is not a clean public image build. Explicit
TF_GLM_NONEXPERT=q4mse, fc7:0.3 startup launcher, context262144, same target
and drafter pins as previous recipe. GLM53_TF_* env inherited from old image
must not be mistaken for implemented features; upstream GLM is serial.
New containers glm53-v060-r0/r1 HTTP8888, rendezvous29651, restart unless-stopped.
Previous glm53-new-r0/r1 stopped, restart disabled, original source preserved.
Rollback: stop both v060 ranks, start old rank1 then rank0 (never simultaneous
full-model residency). 200k rollback containers also preserved.

Verification:33 CPU decisions/cache tests passed; first extension compilation
required more than155 s. Startup estimates90.05GiB within101.36/101.07GiB.
Live full GLM:7/7 decisions, probability invariants, invalid kwargs, tiny
temperature and unchanged chat before/after PASS, repeated after65 s idle PASS.
Mia prompt screen,200 tokens,3 repeats,thinking off: medians prose28.82,
counting66.35,short code44.60tok/s. Warm repeated prompts TTFT~.10–.15s.
No controlled old/new performance A/B, no full262k prompt qualification,
no GPU unit suite this deployment, no batching claim. Source branch PR127
unchanged. Main0.6.0 does not yet contain maintainer's decisions port.
