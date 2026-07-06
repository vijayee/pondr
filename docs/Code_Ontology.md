```python
CODE_ONTOLOGY = {
"classes": {
# ═══════════════════════════════════════════════════════
# CODE STRUCTURE (AST-level artifacts)
# ═══════════════════════════════════════════════════════
"CodeArtifact": {
"subclasses": [
"File", "Module", "Package",
"Class", "Interface", "Trait", "Mixin", "Enum", "Struct",
"Function", "Method", "Constructor", "Destructor",
"Property", "Attribute", "Field",
"Variable", "Constant", "Parameter",
"Type", "Generic", "Union", "Alias",
"Decorator", "Annotation",
"Lambda", "Closure", "Generator",
"Expression", "Statement", "Block",
]
},

        # ═══════════════════════════════════════════════════════
        # VERSION CONTROL
        # ═══════════════════════════════════════════════════════
        "VersionControl": {
            "subclasses": [
                "Commit", "Branch", "Tag", "Release",
                "PullRequest", "MergeRequest", "Patch",
                "Merge", "Rebase", "CherryPick",
                "Conflict", "Diff", "Blame",
                "Fork", "Clone", "Remote",
                "Stash", "Worktree",
            ]
        },

        # ═══════════════════════════════════════════════════════
        # SOFTWARE ENGINEERING CONCEPTS
        # ═══════════════════════════════════════════════════════
        "Issue": {
            "subclasses": [
                "Bug", "Feature", "Enhancement", "Task",
                "TechnicalDebt", "Refactor",
                "PerformanceIssue", "SecurityVulnerability",
                "Regression", "BreakingChange",
                "Deprecation", "Migration",
            ]
        },

        "Test": {
            "subclasses": [
                "UnitTest", "IntegrationTest", "EndToEndTest",
                "PerformanceTest", "SecurityTest", "RegressionTest",
                "Mock", "Stub", "Fixture", "TestSuite",
                "Coverage", "Assertion",
            ]
        },

        "Architecture": {
            "subclasses": [
                "DesignPattern", "ArchitecturalPattern",
                "Component", "Service", "Microservice",
                "Monolith", "Plugin", "Middleware",
                "Layer", "Tier", "Boundary",
                "Adapter", "Facade", "Proxy", "Bridge",
                "Factory", "Singleton", "Observer", "Strategy",
            ]
        },

        "API": {
            "subclasses": [
                "Endpoint", "Route", "Controller",
                "Middleware", "Guard", "Interceptor",
                "Request", "Response", "DTO", "Schema",
                "Query", "Mutation", "Subscription",
                "REST", "GraphQL", "gRPC", "WebSocket",
                "RateLimit", "Throttle", "Cache",
            ]
        },

        "Data": {
            "subclasses": [
                "Database", "Table", "Collection",
                "Column", "Field", "Index",
                "PrimaryKey", "ForeignKey", "Constraint",
                "Query", "Migration", "Seed",
                "Schema", "Model", "Entity", "Relation",
                "Transaction", "Lock", "Deadlock",
                "Cache", "Session", "Connection",
            ]
        },

        "Configuration": {
            "subclasses": [
                "EnvironmentVariable", "ConfigFile",
                "Secret", "Credential", "APIKey",
                "FeatureFlag", "Toggle",
                "Profile", "BuildConfig",
            ]
        },

        # ═══════════════════════════════════════════════════════
        # OPERATIONS / DEVOPS
        # ═══════════════════════════════════════════════════════
        "Infrastructure": {
            "subclasses": [
                "Server", "Container", "Pod", "Cluster",
                "LoadBalancer", "Proxy", "CDN",
                "Database", "Queue", "Cache",
                "Volume", "Network", "Firewall",
                "DNS", "Certificate",
            ]
        },

        "Deployment": {
            "subclasses": [
                "Pipeline", "Stage", "Job", "Step",
                "Build", "Test", "Deploy", "Rollback",
                "Artifact", "Image", "Registry",
                "Environment", "Namespace",
                "HelmChart", "Manifest", "Template",
            ]
        },

        "Observability": {
            "subclasses": [
                "Log", "Metric", "Trace", "Span",
                "Alert", "Incident", "Runbook",
                "Dashboard", "Monitor", "SLO",
                "Error", "Warning", "Debug",
            ]
        },

        # ═══════════════════════════════════════════════════════
        # QUALITY & PROCESS
        # ═══════════════════════════════════════════════════════
        "Quality": {
            "subclasses": [
                "Lint", "Format", "StyleGuide",
                "CodeReview", "Audit",
                "Complexity", "Duplication",
                "Documentation", "README", "Changelog",
                "License", "Dependency",
                "Vulnerability", "CVE", "Patch",
            ]
        },

        "Process": {
            "subclasses": [
                "Sprint", "Milestone", "Roadmap",
                "Estimate", "StoryPoint",
                "Standup", "Retrospective",
                "OnCall", "IncidentResponse",
                "PostMortem", "RCA",
            ]
        },
    },

    # ═══════════════════════════════════════════════════════════
    # PROPERTIES (Relations)
    # ═══════════════════════════════════════════════════════════
    "properties": {

        # ── Code Structure ──
        "contains":         {"domain": "CodeArtifact", "range": "CodeArtifact"},
        "defined_in":       {"domain": "CodeArtifact", "range": "File"},
        "declared_in":      {"domain": "CodeArtifact", "range": "CodeArtifact"},
        "calls":            {"domain": "Function",     "range": "Function"},
        "imports":          {"domain": "File",         "range": "Module"},
        "exports":          {"domain": "Module",       "range": "CodeArtifact"},
        "inherits":         {"domain": "Class",        "range": "Class"},
        "implements":       {"domain": "Class",        "range": "Interface"},
        "overrides":        {"domain": "Method",       "range": "Method"},
        "uses":             {"domain": "CodeArtifact", "range": "CodeArtifact"},
        "instantiates":     {"domain": "CodeArtifact", "range": "Class"},
        "decorates":        {"domain": "Decorator",    "range": "CodeArtifact"},
        "annotates":        {"domain": "Annotation",   "range": "CodeArtifact"},
        "type_of":          {"domain": "Variable",     "range": "Type"},
        "returns":          {"domain": "Function",     "range": "Type"},
        "accepts":          {"domain": "Function",     "range": "Parameter"},
        "raises":           {"domain": "Function",     "range": "Type"},
        "catches":          {"domain": "Block",        "range": "Type"},

        # ── Version Control ──
        "commits":          {"domain": "Branch",       "range": "Commit"},
        "parents":          {"domain": "Commit",       "range": "Commit"},
        "branches_from":    {"domain": "Branch",       "range": "Branch"},
        "merges_into":      {"domain": "Branch",       "range": "Branch"},
        "tags":             {"domain": "Commit",       "range": "Tag"},
        "releases":         {"domain": "Tag",          "range": "Release"},
        "resolves":         {"domain": "Merge",        "range": "Conflict"},
        "cherry_picks":     {"domain": "Commit",       "range": "Commit"},
        "reverts":          {"domain": "Commit",       "range": "Commit"},

        # ── Issue Tracking ──
        "fixes":            {"domain": "Commit",       "range": "Bug"},
        "introduces":       {"domain": "Commit",       "range": "Bug"},
        "regresses":        {"domain": "Commit",       "range": "Regression"},
        "implements":       {"domain": "Commit",       "range": "Feature"},
        "refactors":        {"domain": "Commit",       "range": "Refactor"},
        "addresses":        {"domain": "Commit",       "range": "Issue"},
        "closes":           {"domain": "PullRequest",  "range": "Issue"},
        "blocks":           {"domain": "Issue",        "range": "Issue"},
        "depends_on_issue": {"domain": "Issue",        "range": "Issue"},
        "duplicates":       {"domain": "Issue",        "range": "Issue"},

        # ── Testing ──
        "tests":            {"domain": "Test",         "range": "CodeArtifact"},
        "covers":           {"domain": "TestSuite",    "range": "CodeArtifact"},
        "mocks":            {"domain": "Test",         "range": "CodeArtifact"},
        "asserts":          {"domain": "Test",         "range": "Assertion"},
        "fails":            {"domain": "Test",         "range": "Bug"},
        "regression_tests": {"domain": "RegressionTest","range": "Bug"},

        # ── Architecture & Design ──
        "depends_on":       {"domain": "CodeArtifact", "range": "CodeArtifact"},
        "depends_on_module": {"domain": "Module",       "range": "Module"},
        "depends_on_service":{"domain": "Service",      "range": "Service"},
        "owns":             {"domain": "Team",          "range": "Service"},
        "communicates_with": {"domain": "Service",      "range": "Service"},
        "proxies":          {"domain": "Proxy",         "range": "Service"},
        "balances":         {"domain": "LoadBalancer",  "range": "Service"},
        "caches":           {"domain": "Cache",         "range": "Data"},
        "queues":           {"domain": "Queue",         "range": "Message"},
        "subscribes":       {"domain": "Service",       "range": "Event"},
        "publishes":        {"domain": "Service",       "range": "Event"},

        # ── API ──
        "routes_to":        {"domain": "Route",        "range": "Controller"},
        "handles":          {"domain": "Controller",   "range": "Endpoint"},
        "guards":           {"domain": "Guard",        "range": "Route"},
        "intercepts":       {"domain": "Interceptor",  "range": "Request"},
        "validates":        {"domain": "Middleware",   "range": "Schema"},
        "rate_limits":      {"domain": "RateLimit",    "range": "Endpoint"},
        "authenticates":    {"domain": "Guard",        "range": "Credential"},
        "authorizes":       {"domain": "Guard",        "range": "Permission"},

        # ── Data ──
        "persists":         {"domain": "Repository",   "range": "Entity"},
        "maps_to":          {"domain": "Entity",       "range": "Table"},
        "columns":          {"domain": "Table",        "range": "Column"},
        "references":       {"domain": "ForeignKey",   "range": "PrimaryKey"},
        "indexes":          {"domain": "Index",        "range": "Column"},
        "constrains":       {"domain": "Constraint",   "range": "Column"},
        "migrates":         {"domain": "Migration",    "range": "Schema"},
        "seeds":            {"domain": "Seed",         "range": "Table"},
        "transacts":        {"domain": "Transaction",  "range": "Database"},
        "locks":            {"domain": "Transaction",  "range": "Table"},

        # ── Configuration ──
        "configures":       {"domain": "ConfigFile",   "range": "CodeArtifact"},
        "sets":             {"domain": "EnvironmentVariable", "range": "Value"},
        "secrets":          {"domain": "Secret",       "range": "Credential"},
        "flags":            {"domain": "FeatureFlag",  "range": "Feature"},
        "profiles":         {"domain": "Profile",      "range": "Environment"},

        # ── Deployment ──
        "builds":           {"domain": "Pipeline",     "range": "Artifact"},
        "deploys_to":       {"domain": "Pipeline",     "range": "Environment"},
        "runs_on":          {"domain": "Job",          "range": "Infrastructure"},
        "produces":         {"domain": "Job",          "range": "Artifact"},
        "rolls_back":       {"domain": "Deploy",       "range": "Deploy"},
        "contains_stage":   {"domain": "Pipeline",     "range": "Stage"},
        "contains_job":     {"domain": "Stage",        "range": "Job"},
        "contains_step":    {"domain": "Job",          "range": "Step"},

        # ── Observability ──
        "logs":             {"domain": "CodeArtifact", "range": "Log"},
        "emits":            {"domain": "CodeArtifact", "range": "Metric"},
        "traces":           {"domain": "CodeArtifact", "range": "Span"},
        "alerts_on":        {"domain": "Monitor",      "range": "Metric"},
        "triggers":         {"domain": "Alert",        "range": "Incident"},
        "resolved_by":      {"domain": "Incident",     "range": "Runbook"},
        "caused_by":        {"domain": "Incident",     "range": "Deploy"},
        "postmortem_for":   {"domain": "PostMortem",   "range": "Incident"},

        # ── Quality ──
        "lints":            {"domain": "Lint",         "range": "CodeArtifact"},
        "formats":          {"domain": "Format",       "range": "CodeArtifact"},
        "reviews":          {"domain": "CodeReview",   "range": "PullRequest"},
        "documents":        {"domain": "Documentation","range": "CodeArtifact"},
        "changelogs":       {"domain": "Changelog",    "range": "Release"},
        "depends_on_lib":   {"domain": "Module",       "range": "Dependency"},
        "vulnerable_in":    {"domain": "Vulnerability","range": "Dependency"},
        "patches_vuln":     {"domain": "Patch",        "range": "Vulnerability"},

        # ── Process ──
        "scheduled_in":     {"domain": "Issue",        "range": "Sprint"},
        "milestoned_in":    {"domain": "Issue",        "range": "Milestone"},
        "estimated_at":     {"domain": "Issue",        "range": "Estimate"},
        "discussed_in":     {"domain": "Issue",        "range": "Standup"},
        "retrospected_in":  {"domain": "Sprint",       "range": "Retrospective"},
        "action_item_from": {"domain": "Task",         "range": "Retrospective"},

        # ── Cross-Cutting (Code ↔ Conversation) ──
        "discusses":        {"domain": "Episode",      "range": "CodeArtifact"},
        "modifies":         {"domain": "Episode",      "range": "File"},
        "produces_commit":  {"domain": "Episode",      "range": "Commit"},
        "reviews_code":     {"domain": "Episode",      "range": "PullRequest"},
        "debates":          {"domain": "Episode",      "range": "Issue"},
        "decides_on":       {"domain": "Episode",      "range": "Architecture"},
        "troubleshoots":    {"domain": "Episode",      "range": "Bug"},
        "incident_response":{"domain": "Episode",      "range": "Incident"},
        "pairs_on":         {"domain": "Person",       "range": "CodeArtifact"},
        "owns_code":        {"domain": "Person",       "range": "CodeArtifact"},
        "reviews_work_of":  {"domain": "Person",       "range": "Person"},
    },
}
```