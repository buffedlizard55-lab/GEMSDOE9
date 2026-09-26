# Historical user brief (excerpt retained from previous main README)

This excerpt preserves prior project requirements for context. Other assertions in the old README were untested and were superseded by the current README, measured holdout and audit.

**Project Prompt (Original User Request, Preserved Verbatim for Context)**

> Review the repo. 
> Here are the results from our groups submissions, separated by ....:
> GEMSDOE1 https://buffedlizard55-lab.github.io/GEMSDOE/docs/index.html GEMSDOE SCORE: 0.1563
> https://buffedlizard55-lab.github.io/6GEMSDOE/ 6GEMSDOE SCORE: 0.0286
> GEMSDOE3 https://buffedlizard55-lab.github.io/GEMSDOE3/docs/index.html GEMSDOE3 SCORE: 0.1193
> 1 · SUBMIT FIRST f347b70daa Pindrop nodes
> GEMSDOE2 https://buffedlizard55-lab.github.io/GEMSDOE2/docs/index.html GEMSDOE2 SCORE: 0.1560
> GEMSDOE3 SCORE: 0.0830 2 · SUBMIT SECOND 37f9d5b855 Pindrop catalogue-gap target SECOND SYSTEM
> https://buffedlizard55-lab.github.io/GEMSDOE4/ GEMSDOE 4 SCORE: 0.0343
> GEMSDOE3 SCORE: 0.1152 3 · CONTROL · UPLOAD LAST 4e03fc9705 Pindrop dense ridge control
> https://buffedlizard55-lab.github.io/5GEMSDOE/docs/index.html 5GEMSDOE SCORE: 0.1563
> https://buffedlizard55-lab.github.io/8GEMSDOE/ 8GEMSDOE SCORE: 0.1563
> The leaderboard: https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/
> We need to figure out why we keep scoring 0.1563, are we copying the same work over and over again? we need to come up with different ideas, and not just the same idea tried a different way.
> Need to figure out why 5GEMSDOE and GEMSDOE1 have the same score. We should not be generating the same score submissions, they should all be unique.
> 0.3049 is the highest score right now so we need to design a new strategy, research, testing, analyzing, and generating submission system than the current website. It should be unique, take unique approaches to generating a submission that can score higher than .3049.
> Put this prompt into the repo readme and read it everytime we work on the project as a starting point to make sure we are building what we are aiming for and have a strong base to continue building and improving on making something useful for everyday use. It should solve the problem of having to manually check everything ourselves and having an up to date current feed.
> Review the repo.
> [Core Values and Own the Outcome as focal point]
> Work line by line verifying from official verified trusted sources, provide links for manual review. There should be no manual input, work on your own to complete tasks. Flag any irregularities for review. No hallucinations.
> The goal of this project is to get a full list that follow our requirements. No hallucinations. Verify line by line.
> The goal of this project is to place top of the leaderboard in this competition. Competition: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
> We need to create a project that can compete and place top of the leaderboard. We need to understand the problem, collect all the data and organize it into a clean easily auditable table with official verified links for manual verification.
> Guidelines: https://www.drivendata.org/competitions/306/competition-doe-gems/
> Get familiar with the problem through the overview and problem description, https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/. You might also want to reference additional resources available on the about page, https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/.
> Download the data from the data, https://www.drivendata.org/competitions/306/competition-doe-gems/data/, tab.
> Create and train your own model. This reference solution, https://github.com/drivendataorg/gems-prize-reference-solution implements a simple approach.
> Use your model to generate predictions that match the submission format.
> Tell me what are you limitations and what you need access to during this project. We will need to find free publicly available sources and data from official and verified sources if we are to use 3rd party or external data.
> this pdf outlines how submissions must be entered into the competition. https://docs.nlr.gov/docs/fy26osti/96647.pdf
> You must be able to do your own research, deep research, scientific literature research and organize the knowledge so that we can critically think through the problem and generate a solution through scientific and free publicly available information. this must be done autonomously and must be constantly reviewed and improved upon. Provide suggestions and improvements and implement them.
> No DrivenData auth → cannot auto-download training_features.tif, labels.tif, sample_submission.tif, 1m_DEM_links.csv from https://www.drivendata.org/competitions/306/competition-doe-gems/data/ (verified redirect to login)
> See below for links from the above site. See attached files for links from the above site.
> https://gdr.openei.org/submissions/1391
> Download competition data from https://www.drivendata.org/competitions/306/competition-doe-gems/data/ (requires login) to data/
> See links below for competition data:
> https://www.dropbox.com/scl/fi/aemhtutjgcp6tr3tint94/GEMS_96647.pdf?rlkey=rek210cj2smnmzb8n0sla1vmd&st=wz4kofki&dl=0
> https://www.dropbox.com/scl/fi/6rgvnuady818ol8yqgis4/example_submission.tif?rlkey=kbykilvau066xuogoosbf4cq8&st=8junzdyw&dl=0
> https://www.dropbox.com/scl/fi/t7fyt03qdh9egyme0itwo/existing_faults.tif?rlkey=yiao96uluqdkipf0h5vju71jf&st=rnino7ya&dl=0
> https://www.dropbox.com/scl/fi/3vz9o0wwavi26xaeoxlwr/gems-geodawn-numerical-features.tif?rlkey=je8d8fepqfbst9lnwsq9rkplu&st=zj1lag1r&dl=0
> https://www.dropbox.com/scl/fi/ig0mban712ns1atphgphe/Digital-elevation-model-links-JSON.pdf?rlkey=zm77f1vbtt2if8hlruymptnu3&st=srhhir10&dl=0
> Work line by line verifying from official verified trusted sources, provide links for manual review. There should be no manual input, work on your own to complete tasks. Flag any irregularities for review. No hallucinations.
> Site creation: Create a github page for this repo that has clean ui, user friendly, simple and easy to use. It should be organized and clean. It should include all relevant information in an easy to read format with official verified links as sources for review. Work line by line verify everything no hallucinations.
> The single remaining blocker to training is data placement: run bash scripts/download_competition_data.sh on any unrestricted machine into data/, then python scripts/prepare_data.py — after that the full train→inference→validate pipeline is ready to run (GPU needed for training; metric/losses/validation all verified working here on CPU).
> The site should be able to generate a TIF file that is required for submission. It should be as easy as download to click a File to submit into the competition. This needs to be in the executive summary or the very beginning of the site. it should be obvious when you visit the site.
> I tried to submit the document that i downloaded from the site but it returned this error on the submission form: "Predicted values must be in range [0, 1]"
> Also we need to give it a unique name and A short comment to help you or your team tell submissions apart later e.g. clustering with k=25
> Here is the submission page when i click submit file: New submission File to submit No file chosen You can submit a single-band GeoTIFF (.tif) file, or a .zip file containing a single GeoTIFF, with your predictions. It must match the submission format's CRS, shape, and geotransform. You may wish to review the competition rules first. Note (optional) A short comment to help you or your team tell submissions apart later e.g. clustering with k=25
> Create a executive summary subpage that explains exactly how to make a submission into the contest.
> Work on the next steps from the previous sessions first.
> [Repeated core values and verification requirements]

