# OpenClaw on AWS — Public Endpoints

## Control UI (Gateway)

| Protocol | URL |
|----------|-----|
| HTTPS (CloudFront, Amazon CA) | `https://d1h89povp972z9.cloudfront.net` |
| Tokenized | `https://d1h89povp972z9.cloudfront.net/#token=6bc692ba72f1e41a3eacc4fd76961fbed26a5a9bac65cdbb` |
| HTTPS (NLB, deprecated) | `https://openclaw-nlb-b58d647a7c5feacf.elb.us-west-2.amazonaws.com` |
| SSM Localhost | `http://localhost:18789/#token=6bc692ba72f1e41a3eacc4fd76961fbed26a5a9bac65cdbb` |

## A2A Protocol (Agent)

### HTTPS — CloudFront with Amazon CA cert (HTTP/2 + HTTP/3)

| Endpoint | URL |
|----------|-----|
| Agent Card | `https://d1h89povp972z9.cloudfront.net/a2a/.well-known/agent-card.json` |
| Agent Card (alt) | `https://d1h89povp972z9.cloudfront.net/.well-known/agent-card.json` |
| JSON-RPC | `https://d1h89povp972z9.cloudfront.net/a2a` |
| Streaming | `https://d1h89povp972z9.cloudfront.net/a2a/stream` |

### HTTP — ALB with WAF (US-only geo-filter)

| Endpoint | URL |
|----------|-----|
| Agent Card | `http://openclaw-agent-alb-1917535062.us-west-2.elb.amazonaws.com/.well-known/agent-card.json` |
| JSON-RPC | `http://openclaw-agent-alb-1917535062.us-west-2.elb.amazonaws.com/a2a` |
| Streaming | `http://openclaw-agent-alb-1917535062.us-west-2.elb.amazonaws.com/a2a/stream` |

### HTTPS — NLB with self-signed cert (deprecated, replaced by CloudFront)

| Endpoint | URL |
|----------|-----|
| Agent Card | `https://openclaw-nlb-b58d647a7c5feacf.elb.us-west-2.amazonaws.com:8443/.well-known/agent-card.json` |
| JSON-RPC | `https://openclaw-nlb-b58d647a7c5feacf.elb.us-west-2.amazonaws.com:8443/a2a` |
| Streaming | `https://openclaw-nlb-b58d647a7c5feacf.elb.us-west-2.amazonaws.com:8443/a2a/stream` |

### VPC Internal

| Endpoint | URL |
|----------|-----|
| Agent Card | `http://10.0.1.74:8080/.well-known/agent-card.json` |
| JSON-RPC | `http://10.0.1.74:8080/a2a` |
| Streaming | `http://10.0.1.74:8080/a2a/stream` |

## Infrastructure

| Resource | Value |
|----------|-------|
| AWS Account | `730774079533` |
| Region | `us-west-2` |
| EC2 Instance | `i-0ee108009eaa752a7` (t4g.medium, arm64) |
| IAM Role | `Bedrock_openclaw_role` |
| Model | `amazon-bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0` |
| VPC | `vpc-0b37803cd29420dd7` |
| ALB (HTTP+WAF) | `openclaw-agent-alb-1917535062.us-west-2.elb.amazonaws.com` |
| NLB (TCP/TLS, deprecated) | `openclaw-nlb-b58d647a7c5feacf.elb.us-west-2.amazonaws.com` |
| CloudFront (Amazon CA) | `d1h89povp972z9.cloudfront.net` (Distribution: `E3PPF5VA365DLI`) |
| CloudFormation Stack | `openclaw-bedrock` |

## SSM Access

```bash
# Port forward gateway
aws ssm start-session --target i-0ee108009eaa752a7 --region us-west-2 \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["18789"],"localPortNumber":["18789"]}'

# Port forward agent
aws ssm start-session --target i-0ee108009eaa752a7 --region us-west-2 \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["8080"],"localPortNumber":["8080"]}'

# Shell access
aws ssm start-session --target i-0ee108009eaa752a7 --region us-west-2
```

## Java Client Example

```java
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.regex.Pattern;

public class A2AClient {

    private static final String BASE = "https://d1h89povp972z9.cloudfront.net";
    private static final HttpClient client = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_2)
            .connectTimeout(Duration.ofSeconds(10))
            .build();

    private static final Pattern TASK_ID = Pattern.compile(
            "\"task\":\\s*\\{\\s*\"id\":\\s*\"([^\"]+)\"");

    /** Fetch the agent card. */
    public static String getAgentCard() throws Exception {
        var req = HttpRequest.newBuilder()
                .uri(URI.create(BASE + "/a2a/.well-known/agent-card.json"))
                .GET().build();
        return client.send(req, HttpResponse.BodyHandlers.ofString()).body();
    }

    /** Send a JSON-RPC request to /a2a and return the response body. */
    private static String rpc(String method, String paramsJson) throws Exception {
        var body = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"%s\",\"params\":%s}"
                .formatted(method, paramsJson);
        var req = HttpRequest.newBuilder()
                .uri(URI.create(BASE + "/a2a"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();
        return client.send(req, HttpResponse.BodyHandlers.ofString()).body();
    }

    /** Send a message (message/send or message/stream) and return the task ID. */
    public static String sendMessage(String text, boolean streaming) throws Exception {
        var method = streaming ? "message/stream" : "message/send";
        var params = "{\"message\":{\"role\":\"user\",\"parts\":[{\"kind\":\"text\",\"text\":\"%s\"}]}}"
                .formatted(text);
        var resp = rpc(method, params);
        var m = TASK_ID.matcher(resp);
        if (!m.find()) throw new RuntimeException("No task ID in response: " + resp);
        return m.group(1);
    }

    /** Poll tasks/get until completed or failed. */
    public static String pollTask(String taskId) throws Exception {
        var params = "{\"id\":\"%s\",\"stateTransitionHistory\":true}".formatted(taskId);
        for (int i = 0; i < 20; i++) {
            Thread.sleep(3000);
            var resp = rpc("tasks/get", params);
            if (resp.contains("TASK_STATE_COMPLETED") || resp.contains("TASK_STATE_FAILED")) {
                return resp;
            }
        }
        throw new RuntimeException("Task timed out: " + taskId);
    }

    public static void main(String[] args) throws Exception {
        // 1. Discover agent
        System.out.println("Agent Card: " + getAgentCard());

        // 2. message/send
        var taskId = sendMessage("What is 2+2? Reply with just the number.", false);
        System.out.println("Task ID: " + taskId);

        // 3. Poll for result
        System.out.println("Result: " + pollTask(taskId));

        // 4. message/stream (same flow, different method)
        var streamTaskId = sendMessage("What is 3+3? Reply with just the number.", true);
        System.out.println("Stream Result: " + pollTask(streamTaskId));
    }
}
```
