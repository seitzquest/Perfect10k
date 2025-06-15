CLOUDFLARE_API_TOKEN=
DNS_RECORD_NAME="seitzquest.com"

# Get Zone ID for the domain
# CLOUDFLARE_API_TOKEN='xKh_EUtJ7r1cvifZVf12Xso3r4B_yg6qlxcAp_yO'
# curl -X GET "https://api.cloudflare.com/client/v4/zones?name=seitzquest.com" \
#      -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN"
ZONE_ID='de98dfce25ee54c9702e0696f8e9ea04'

# Get DNS records for the domain
# curl -X GET "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records?name=$DNS_RECORD_NAME" \
#      -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN"
# curl -X GET "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records?name=www.$DNS_RECORD_NAME" \
#      -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN"
RECORD_ID='90dda558fa3269e17e847f29cab79231'
RECORD_ID_WWW='8bf95dae68d18f93121de11999d878f6'


CURRENT_IP_ADDRESS="\"$(curl -s -4 ip.me)\""
CURRENT_DNS_IP=$(curl -sX GET "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records/${RECORD_ID}" -H "Content-Type:application/json" -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" | jq '.result["content"]')
echo "Current DNS IP: $CURRENT_DNS_IP"
if [ "$CURRENT_IP_ADDRESS" != "$CURRENT_DNS_IP" ]; then
    echo "Updating DNS record for $DNS_RECORD_NAME to $CURRENT_IP_ADDRESS"
    curl -sX PUT "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records/${RECORD_ID}" \
        -H "Content-Type:application/json" \
        -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
        --data "{\"type\":\"A\",\"name\":\"${DNS_RECORD_NAME}\",\"content\":${CURRENT_IP_ADDRESS},\"proxied\":true}" \
        | jq '.'
    curl -sX PUT "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records/${RECORD_ID_WWW}" \
        -H "Content-Type:application/json" \
        -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
        --data "{\"type\":\"A\",\"name\":\"www.${DNS_RECORD_NAME}\",\"content\":${CURRENT_IP_ADDRESS},\"proxied\":true}" \
        | jq '.'
    echo "DNS record updated successfully."
else
    echo "DNS record is already up-to-date."
fi