import React from "react";
import { PageSwitcher } from "../PageSwitcher/PageSwitcher";
import { AppProvider, Card, Page, Link } from "@shopify/polaris";
import enTranslations from "@shopify/polaris/locales/en.json";

const translations = {
  "title": "FaceOff - Latent GAN Space Brain Surfer",
  "subtitle": [
    "FaceOff is an interactive art piece focused on the relationship between real and artificial brains. ",
    "You will use your webcam to take in image of your face, and allow an artificial intelligence to find a digital match, a barcode that matches your face. ",
    "You will then use your own brain waves to make slight changes in this digital barcode, and watch as your face morphs before your eyes. "

  ],
  "footer": "FaceOff - Latent GAN Space Brain Surfer with the Muse brought to you by Mathewson Sons. "
}


export function App() {
  return (
    <AppProvider i18n={enTranslations}>
      <Page title={translations.title} subtitle={translations.subtitle}>
        <PageSwitcher />
        <Card sectioned>
          <p>{translations.footer}
          A  
          <Link url="http://kylemathewson.com"> Ky</Link>
          <Link url="http://korymathewson.com">Kor</Link>
          <Link url="http://keyfer.ca">Key </Link>
          Production.
          <Link url="https://github.com/kylemath/faceoff/"> Github source code </Link>
        </p>
        </Card>
      </Page>
    </AppProvider>
  );
}


