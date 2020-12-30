import React, { useState, useCallback } from "react";
import { MuseClient } from "muse-js";
import { Card, Stack, Button, ButtonGroup, Checkbox } from "@shopify/polaris";

import { mockMuseEEG } from "./utils/mockMuseEEG";
import * as generalTranslations from "./components/translations/en";
import { emptyAuxChannelData } from "./components/chartOptions";

import * as funSpectro from "./components/EEGEduSpectro";

window.firstAnimate = true;

export function PageSwitcher() {

  // For auxEnable settings
  const [checked, setChecked] = useState(false);
  const handleChange = useCallback((newChecked) => setChecked(newChecked), []);
  window.enableAux = checked;
  if (window.enableAux) {
    window.nchans = 5;
  } else {
    window.nchans = 4;
  }
  let showAux = false; // if it is even available to press (to prevent in some modules)

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

  // Render the entire page using above functions
  return (
    <React.Fragment>
      <funSpectro.renderModule data={spectroData}/>
      <Card sectioned>
        <Stack>
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
          <Checkbox
            label="Enable Muse Auxillary Channel"
            checked={checked}
            onChange={handleChange}
            disabled={!showAux || status !== generalTranslations.connect}
          />
        </Stack>
      </Card>
      {funSpectro.renderSliders(setSpectroData, setSpectroSettings, status, spectroSettings)}
    </React.Fragment>
  );
}
