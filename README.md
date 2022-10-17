# IBL public website prototype

## Development notes

* Install Python requirements
* Put the data in the `data/` subdirectory. Each session should be in a separate folder which name should be the insertion's uuid.
* Launch the development server with `python flaskapp.py` (or `./run.sh`)
* Go to `http://localhost:4321/`


## Deployment on a production server

* Tested on Ubuntu 20.04+
* Create a Python virtual env
* `pip install -r requirements.txt`
* `sudo nano /etc/systemd/system/flaskapp.service` and put:

```
[Unit]
Description=IBL website
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/website/
Environment="PATH=/home/ubuntu/website/bin"
ExecStart=sudo /home/ubuntu/website/bin/python flaskapp.py --port 80

[Install]
WantedBy=multi-user.target
```

## Unity dev notes

### Unity -> Javascript link

You can add javascript functions to access in this `.jslib` file: [unity_js_link](https://github.com/int-brain-lab/website/blob/main/UnityMiniBrainClient/Assets/Plugins/unity_js_link.jslib). These functions can be called from anywhere in Unity by including a DLL import call referencing the corresponding Javascript functions. Note that **only individual strings or numerical types** can be passed to javascript without dealing directly with the javascript heap.

```
[DllImport("__Internal")]
private static extern void SelectPID(string pid);
```

### Javascript -> Unity link

We exposed a javascript variable `myGameInstance` which can be used to call arbitrary Unity code by using the `SendMessage` function. Floats and strings can be passed as variables.

```
myGameInstance.SendMessage('MyGameObject', 'MyFunction');
myGameInstance.SendMessage('MyGameObject', 'MyFunction', 5);
myGameInstance.SendMessage('MyGameObject', 'MyFunction', 'string');
```

### TrialViewer Data

The video files and the .bytes/.csv files used by the trial viewer are generated by the Python code in the TrialViewer/pipelines folder. The video files produced in the /final/ folder should be copied onto the server in the folder `/var/www/ibl_website/trialviewer_data/WebGL/` the .bytes/.csv files created in the pipelines/final folder need to be copied into the TrialViewer AddressableAssets folder. Each folder should then be assigned to an independent addressables group. The built bundles need to be deployed along with the remote catalog to the folder `/var/www/ibl_website/trialviewer_data/WebGL/` next to the videos. Note that the remote catalog is hard-coded into the code and needs to be updated if any changes are made.

### Build to WebGL

To host the built website you build either the IBLMini or the TrialViewer build, and then copy the compressed build files to the corresponding folder on the server, look in `/var/www/ibl_website/`
