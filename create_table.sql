-- 用户行为表
CREATE TABLE IF NOT EXISTS `user_behavior` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `user_id` bigint NOT NULL,
  `service_id` bigint NOT NULL,
  `behavior_type` varchar(20) NOT NULL COMMENT '行为类型：view-浏览, favorite-收藏, click-点击',
  `duration` int DEFAULT NULL COMMENT '停留时间（秒）',
  `extra_data` text COMMENT '额外信息，JSON格式',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_service_id` (`service_id`),
  KEY `idx_behavior_type` (`behavior_type`),
  KEY `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='用户行为表';

-- 用户标签表
CREATE TABLE IF NOT EXISTS `user_tag` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `user_id` bigint NOT NULL,
  `tag` varchar(50) NOT NULL COMMENT '标签名称',
  `weight` decimal(5,2) NOT NULL DEFAULT '1.00' COMMENT '标签权重',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_user_tag` (`user_id`,`tag`),
  KEY `idx_user_id` (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='用户标签表';

-- 推荐记录表
CREATE TABLE IF NOT EXISTS `recommendation_log` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `user_id` bigint NOT NULL,
  `service_id` bigint NOT NULL,
  `score` decimal(5,2) NOT NULL COMMENT '推荐分数',
  `is_clicked` tinyint(1) DEFAULT '0' COMMENT '是否被点击',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_service_id` (`service_id`),
  KEY `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='推荐记录表'; 