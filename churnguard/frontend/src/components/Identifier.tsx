import Box from "@cloudscape-design/components/box";
import CopyToClipboard from "@cloudscape-design/components/copy-to-clipboard";
import SpaceBetween from "@cloudscape-design/components/space-between";

interface Props {
  label: string;
  value: string | null | undefined;
}

// Renders a real resource identifier (ARN, endpoint name, S3 URI, SNS topic ARN)
// with a Cloudscape CopyToClipboard control. Value is always a real backend-
// supplied string; a missing value renders an explicit em dash, not a guess.
export default function Identifier({ label, value }: Props) {
  return (
    <SpaceBetween size="xxs">
      <Box variant="awsui-key-label">{label}</Box>
      {value ? (
        <CopyToClipboard
          variant="inline"
          textToCopy={value}
          copyButtonAriaLabel={`Copy ${label}`}
          copySuccessText="Copied"
          copyErrorText="Copy failed"
        />
      ) : (
        <Box variant="p" color="text-body-secondary">
          —
        </Box>
      )}
    </SpaceBetween>
  );
}
