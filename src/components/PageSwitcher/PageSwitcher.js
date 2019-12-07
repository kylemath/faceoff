import React, { useState, useCallback } from "react";

import { Select, Card } from "@shopify/polaris";

import EEGEduRaw from "./components/EEGEduRaw/EEGEduRaw";
import EEGEduSpectra from "./components/EEGEduSpectra/EEGEduSpectra";

import * as translations from "./translations/en.json";

export function PageSwitcher() {
  const [selected, setSelected] = useState(translations.types.raw);
  const handleSelectChange = useCallback(value => setSelected(value), []);
  const options = [
    { label: translations.types.raw, value: translations.types.raw },
    { label: translations.types.spectra, value: translations.types.spectra }
  ];

  function renderCharts() {
    switch (selected) {
      case translations.types.raw:
        return <EEGEduRaw />;
      default:
        return <EEGEduSpectra />;
    }
  }

  return (
    <React.Fragment>
      <Card title={translations.title} sectioned>
        <Select
          label={""}
          options={options}
          onChange={handleSelectChange}
          value={selected}
        />
      </Card>

      {renderCharts()}
    </React.Fragment>
  );
}