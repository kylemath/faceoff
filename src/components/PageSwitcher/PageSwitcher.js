import React, { useState } from "react";
import { MuseClient } from "muse-js";
import { Card, Stack, Button, ButtonGroup } from "@shopify/polaris";

import { mockMuseEEG } from "./utils/mockMuseEEG";
import * as generalTranslations from "./components/translations/en";
import { emptyAuxChannelData } from "./components/chartOptions";

import * as funSpectro from "./components/EEGEduSpectro";

window.firstAnimate = true;

export function PageSwitcher() {


  window.nchans = 4;
  // data pulled out of multicast$
  const [spectroData, setSpectroData] = useState(emptyAuxChannelData);

  // pipe settings
  const [spectroSettings, setSpectroSettings] = useState(funSpectro.getSettings);

  // connection status
  const [status, setStatus] = useState(generalTranslations.connect);

   async function connect() {
    try {
      if (window.debugWithMock) {
        // Debug with Mock EEG Data
        setStatus(generalTranslations.connectingMock);
        window.source = {};
        window.source.connectionStatus = {};
        window.source.connectionStatus.value = true;
        window.source.eegReadings$ = mockMuseEEG(256);
        setStatus(generalTranslations.connectedMock);
      } else {
        // Connect with the Muse EEG Client
        setStatus(generalTranslations.connecting);
        window.source = new MuseClient();
        window.source.enableAux = window.enableAux;
        await window.source.connect();
        await window.source.start();
        window.source.eegReadings$ = window.source.eegReadings;
        setStatus(generalTranslations.connected);
      }
      if (
        window.source.connectionStatus.value === true &&
        window.source.eegReadings$
      ) {

        funSpectro.buildPipe(spectroSettings);
        funSpectro.setup(setSpectroData, spectroSettings);

      }
    } catch (err) {
      setStatus(generalTranslations.connect);
      console.log("Connection error: " + err);
    }
  }

  function refreshPage(){
    window.location.reload();
  }

  const strings = {
    "museOn": [
      "If you do not have a Muse headband you can click the Mock Data button to use simluated data. ",
      "The first step will be to turn on your Muse headband and click the connect button. ",
      "This will open a screen and will list available Muse devices. ",
      "Select the serial number written on your Muse. ",
      "The data will begin streaming from your brain, which we will use in a few steps."
    ]
  }

  // Render the entire page using above functions
  return (
    <React.Fragment>
      <Card sectioned>
        <Stack>
          <p>{strings.museOn}</p>
          <ButtonGroup>
            <Button
              primary={status === generalTranslations.connect}
              disabled={status !== generalTranslations.connect}
              onClick={() => {
                window.debugWithMock = false;
                connect();
              }}
            >
              {status}
            </Button>
            <Button
              disabled={status !== generalTranslations.connect}
              onClick={() => {
                window.debugWithMock = true;
                connect();
      
              }}
            >
              {status === generalTranslations.connect ? generalTranslations.connectMock : status}
            </Button>
            <Button
              destructive
              onClick={refreshPage}
              primary={status !== generalTranslations.connect}
              disabled={status === generalTranslations.connect}
            >
              {generalTranslations.disconnect}
            </Button>
          </ButtonGroup>
        </Stack>
      </Card>
      <funSpectro.RenderModule data={spectroData}/>
      {funSpectro.renderSliders(setSpectroData, setSpectroSettings, status, spectroSettings)}
    </React.Fragment>
  );
}
