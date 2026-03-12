use chrono::{DateTime, Local, Utc};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct CurrentTimeResult {
    pub date: String,
    pub time: String,
    pub timezone: String,
    pub timestamp: i64,
    pub iso8601: String,
}

pub fn get_current_time(tz: Option<&str>) -> CurrentTimeResult {
    match tz {
        Some("local") | None => {
            let now_local: DateTime<Local> = Local::now();
            CurrentTimeResult {
                date: now_local.format("%Y-%m-%d").to_string(),
                time: now_local.format("%H:%M:%S").to_string(),
                timezone: "Local".to_string(),
                timestamp: now_local.timestamp(),
                iso8601: now_local.to_rfc3339(),
            }
        }
        Some(tz_str) => {
            let now_utc: DateTime<Utc> = Utc::now();
            CurrentTimeResult {
                date: now_utc.format("%Y-%m-%d").to_string(),
                time: now_utc.format("%H:%M:%S").to_string(),
                timezone: tz_str.to_string(),
                timestamp: now_utc.timestamp(),
                iso8601: now_utc.to_rfc3339(),
            }
        }
    }
}
