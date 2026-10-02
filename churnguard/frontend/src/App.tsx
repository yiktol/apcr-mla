import { useCallback, useEffect, useState } from "react";
import AppLayout from "@cloudscape-design/components/app-layout";
import ContentLayout from "@cloudscape-design/components/content-layout";
import Header from "@cloudscape-design/components/header";
import Tabs from "@cloudscape-design/components/tabs";
import { api } from "./api/client";
import type { RegistryResponse } from "./api/types";
import LandingPage from "./tabs/LandingPage";
import LiveScoringTab from "./tabs/LiveScoringTab";
import BatchTab from "./tabs/BatchTab";
import AsyncTab from "./tabs/AsyncTab";
import ServerlessTab from "./tabs/ServerlessTab";
import PipelineRegistryTab from "./tabs/PipelineRegistryTab";
import BlueGreenTab from "./tabs/BlueGreenTab";
import MultiModelTab from "./tabs/MultiModelTab";

export default function App() {
  // Registry state is lifted so tab 5 (approve) and tab 6 (green candidate) share
  // the same real registry data — the explicit continuous flow across tabs.
  const [registry, setRegistry] = useState<RegistryResponse | null>(null);

  const refreshRegistry = useCallback(async () => {
    try {
      setRegistry(await api.getRegistry());
    } catch {
      // The registry panel surfaces its own errors; a background refresh failure
      // must not crash the shell.
    }
  }, []);

  useEffect(() => {
    refreshRegistry();
  }, [refreshRegistry]);

  return (
    <AppLayout
      navigationHide
      toolsHide
      content={
        <ContentLayout header={<Header variant="h1">ChurnGuard — SageMaker deployment demo</Header>}>
          <Tabs
            tabs={[
              { id: "home", label: "Overview", content: <LandingPage /> },
              { id: "realtime", label: "Live scoring", content: <LiveScoringTab /> },
              { id: "batch", label: "Bulk / Batch", content: <BatchTab /> },
              { id: "async", label: "Async report", content: <AsyncTab /> },
              { id: "serverless", label: "Serverless", content: <ServerlessTab /> },
              {
                id: "pipeline",
                label: "Pipeline & Registry",
                content: (
                  <PipelineRegistryTab registry={registry} refreshRegistry={refreshRegistry} />
                ),
              },
              {
                id: "bluegreen",
                label: "Blue/Green",
                content: <BlueGreenTab registry={registry} />,
              },
              { id: "mme", label: "Hosting (Multi-Model)", content: <MultiModelTab /> },
            ]}
          />
        </ContentLayout>
      }
    />
  );
}
