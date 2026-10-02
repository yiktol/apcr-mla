import Alert from "@cloudscape-design/components/alert";
import Box from "@cloudscape-design/components/box";
import { ApiError } from "../api/client";

interface Props {
  error: unknown;
}

// Surfaces the backend error envelope (code + message + aws_request_id) verbatim
// so the operator sees the real AWS failure, never a glossed-over message.
export default function ErrorAlert({ error }: Props) {
  if (!error) return null;

  if (error instanceof ApiError) {
    return (
      <Alert type="error" header={`${error.code} (HTTP ${error.httpStatus || "network"})`}>
        <Box variant="p">{error.message}</Box>
        {error.awsRequestId && (
          <Box variant="small" color="text-body-secondary">
            aws_request_id: {error.awsRequestId}
          </Box>
        )}
      </Alert>
    );
  }

  return (
    <Alert type="error" header="Error">
      {(error as Error).message ?? String(error)}
    </Alert>
  );
}
